"""Playwright tool wrappers and LLM-facing tool metadata.

This module provides:
- a small `PlaywrightController` to manage a browser and page (sync API),
- core tool functions implementing common browser actions,
- modular tool definitions from services like YouTube, Spotify, etc.,
- `get_tool_specs()` which returns JSON-schema-like metadata for each tool
  (useful for registering tools with an LLM), and
- `execute_tool(name, args)` as a single entrypoint the LLM can call.

Design notes (flow):
- The LLM should be given the output of `get_tool_specs()` so it knows
  available tools and parameter schemas.
- The LLM calls `execute_tool(tool_name, args_dict)` with a single JSON
  object of arguments that match the tool's schema.
- The tool returns a JSON-serializable dict with `ok` and `result` or
  `error` fields.

Security: validate inputs, restrict arbitrary script execution, and run
Playwright in a sandboxed environment where possible. Avoid exposing
secrets to the page context or returning sensitive data.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import json

from util_classes import _result_error, _result_ok, ToolSpec, ToolDefinition
from YoutubeController import PlaywrightController

from tools_youtube import youtube_tools

try:
	from playwright.sync_api import sync_playwright, Page, Browser
except Exception:
	# If Playwright isn't installed in the environment, keep module importable
	sync_playwright = None
	Page = Any
	Browser = Any

_controller = PlaywrightController()


# ============================================================================
# CORE TOOL FUNCTIONS
# ============================================================================


def tool_start_browser(_controller: PlaywrightController, args: Dict[str, Any]) -> Dict[str, Any]:
	headless = bool(args.get("headless", True))
	try:
		_controller.start(headless=headless)
		return _result_ok({"status": "browser started", "headless": headless})
	except Exception as e:
		return _result_error(e)


def tool_goto(_controller: PlaywrightController, args: Dict[str, Any]) -> Dict[str, Any]:
	url = args.get("url")
	if not url:
		return _result_error("'url' parameter is required")
	try:
		page = _controller.page
		if page is None:
			return _result_error("browser not started")
		timeout = args.get("timeout", 30000)
		page.goto(url, timeout=timeout)
		return _result_ok({"status": "navigated", "url": url})
	except Exception as e:
		return _result_error(e)


def tool_click(_controller: PlaywrightController, args: Dict[str, Any]) -> Dict[str, Any]:
	selector = args.get("selector")
	if not selector:
		return _result_error("'selector' parameter is required")
	try:
		page = _controller.page
		if page is None:
			return _result_error("browser not started")
		page.click(selector)
		return _result_ok({"status": "clicked", "selector": selector})
	except Exception as e:
		return _result_error(e)


def tool_fill(_controller: PlaywrightController, args: Dict[str, Any]) -> Dict[str, Any]:
	selector = args.get("selector")
	value = args.get("value", "")
	if not selector:
		return _result_error("'selector' parameter is required")
	try:
		page = _controller.page
		if page is None:
			return _result_error("browser not started")
		page.fill(selector, value)
		return _result_ok({"status": "filled", "selector": selector})
	except Exception as e:
		return _result_error(e)


def tool_screenshot(_controller: PlaywrightController, args: Dict[str, Any]) -> Dict[str, Any]:
	path = args.get("path")
	if not path:
		return _result_error("'path' parameter is required")
	try:
		page = _controller.page
		if page is None:
			return _result_error("browser not started")
		page.screenshot(path=path, full_page=bool(args.get("full_page", False)))
		return _result_ok({"status": "screenshot saved", "path": path})
	except Exception as e:
		return _result_error(e)


def tool_eval(_controller: PlaywrightController, args: Dict[str, Any]) -> Dict[str, Any]:
	script = args.get("script")
	if not script:
		return _result_error("'script' parameter is required")
	# Security: do not allow arbitrary access to system resources from page scripts
	try:
		page = _controller.page
		if page is None:
			return _result_error("browser not started")
		value = page.evaluate(script)
		return _result_ok({"value": value})
	except Exception as e:
		return _result_error(e)


def tool_text_content(_controller: PlaywrightController, args: Dict[str, Any]) -> Dict[str, Any]:
	selector = args.get("selector")
	if not selector:
		return _result_error("'selector' parameter is required")
	try:
		page = _controller.page
		if page is None:
			return _result_error("browser not started")
		text = page.text_content(selector)
		return _result_ok({"text": text})
	except Exception as e:
		return _result_error(e)


def tool_close_browser(_controller: PlaywrightController, args: Dict[str, Any]) -> Dict[str, Any]:
	try:
		_controller.stop()
		return _result_ok({"status": "browser closed"})
	except Exception as e:
		return _result_error(e)


def tool_find_element(_controller: PlaywrightController, args: Dict[str, Any]) -> Dict[str, Any]:
	"""Find an element heuristically based on a human-friendly query string.

	Heuristics used (in order of weight): class name contains token, id contains token,
	visible text contains token, attributes (aria-label/title/alt) contain token.

	Returns a dict with `selector`, `outerHTML`, `text`, and a `score`.
	"""
	query = args.get("query")
	if not query:
		return _result_error("'query' parameter is required")
	try:
		page = _controller.page
		if page is None:
			return _result_error("browser not started")

		q = query.strip().lower()
		tokens = [t for t in q.split() if t]

		# Limit how many nodes to inspect to avoid long loops
		nodes = page.query_selector_all("body *")
		best = None
		best_score = -1

		for idx, node in enumerate(nodes):
			if idx >= 1000:
				break
			try:
				cls = (node.get_attribute("class") or "").lower()
				id_attr = (node.get_attribute("id") or "").lower()
				aria = (node.get_attribute("aria-label") or "").lower()
				title = (node.get_attribute("title") or "").lower()
				alt = (node.get_attribute("alt") or "").lower()
				role = (node.get_attribute("role") or "").lower()
				text = (node.text_content() or "").strip()
				text_l = text.lower()

				score = 0
				for t in tokens:
					if t and t in cls:
						score += 30
					if t and t in id_attr:
						score += 25
					if t and t in text_l:
						score += 20
					if t and (t in aria or t in title or t in alt):
						score += 15
					if t and t == role:
						score += 5

				# small boost for clickable elements
				tag = node.evaluate("el => el.tagName.toLowerCase()")
				if tag in ("button", "a", "input"):
					score += 2

				if score > best_score:
					# compute a CSS selector and outerHTML via JS for precision
					info = page.evaluate(
						"el => { function cssPath(el){ if(el.id) return '#'+el.id; var path=[]; while(el && el.nodeType===1){ var sel=el.nodeName.toLowerCase(); if(el.className){ var c=String(el.className).trim().split(/\s+/)[0]; if(c) sel += '.'+c; } var parent=el.parentElement; if(parent){ var sameTagSiblings = Array.from(parent.children).filter(e=>e.nodeName===el.nodeName); if(sameTagSiblings.length>1){ var i=Array.from(parent.children).filter(e=>e.nodeName===el.nodeName).indexOf(el)+1; sel += ':nth-of-type('+i+')'; } } path.unshift(sel); el=el.parentElement; } return path.join(' > ');} return {selector: cssPath(el), outerHTML: el.outerHTML, text: el.innerText}; }",
						node,
					)
					best = {"score": score, "selector": info.get("selector"), "outerHTML": info.get("outerHTML"), "text": info.get("text")}
					best_score = score
			except Exception:
				continue

		if best is None:
			return _result_ok({"found": False, "query": query})
		return _result_ok({"found": True, "query": query, "match": best})
	except Exception as e:
		return _result_error(e)


# ============================================================================
# TOOL REGISTRY: Build the _TOOLS dict from core and modular tools
# ============================================================================

_CORE_TOOLS = {
	"start_browser": tool_start_browser,
	"goto": tool_goto,
	"click": tool_click,
	"fill": tool_fill,
	"screenshot": tool_screenshot,
	"eval": tool_eval,
	"text_content": tool_text_content,
	"close_browser": tool_close_browser,
	"find_element": tool_find_element,
}

# Register modular tools (YouTube, Spotify, etc.)
_MODULAR_TOOLS = {}
for tool_def in youtube_tools:
	_MODULAR_TOOLS[tool_def.name] = tool_def.func

_TOOLS: Dict[str, Callable[[PlaywrightController, Dict[str, Any]], Dict[str, Any]]] = {**_CORE_TOOLS, **_MODULAR_TOOLS}


# ============================================================================
# CORE TOOL SPECS (if any tools are added, add specs here)
# ============================================================================

_CORE_TOOL_SPECS = {
	"start_browser": ToolSpec(
		name="start_browser",
		description="Launch a browser instance. Args: headless (bool).",
		parameters={"type": "object", "properties": {"headless": {"type": "boolean"}}},
	),
	"goto": ToolSpec(
		name="goto",
		description="Navigate the current page to a URL. Args: url (string).",
		parameters={"type": "object", "properties": {"url": {"type": "string"}, "timeout": {"type": "integer"}}},
	),
	"click": ToolSpec(
		name="click",
		description="Click an element specified by a CSS selector.",
		parameters={"type": "object", "properties": {"selector": {"type": "string"}}},
	),
	"fill": ToolSpec(
		name="fill",
		description="Fill an input identified by selector with value.",
		parameters={"type": "object", "properties": {"selector": {"type": "string"}, "value": {"type": "string"}}},
	),
	"screenshot": ToolSpec(
		name="screenshot",
		description="Save a screenshot to 'path'.",
		parameters={"type": "object", "properties": {"path": {"type": "string"}, "full_page": {"type": "boolean"}}},
	),
	"eval": ToolSpec(
		name="eval",
		description="Evaluate a small JS snippet in page context and return the result.",
		parameters={"type": "object", "properties": {"script": {"type": "string"}}},
	),
	"text_content": ToolSpec(
		name="text_content",
		description="Return the textContent of the first element matching selector.",
		parameters={"type": "object", "properties": {"selector": {"type": "string"}}},
	),
	"close_browser": ToolSpec(
		name="close_browser",
		description="Close the browser and free resources.",
		parameters={"type": "object", "properties": {}},
	),
	"find_element": ToolSpec(
		name="find_element",
		description="Heuristically find an element by a human query. Args: query (string).",
		parameters={"type": "object", "properties": {"query": {"type": "string"}}},
	),
}


def get_tool_specs() -> Dict[str, Any]:
	"""Return metadata for each tool suitable for LLM tool registration.

	Each entry contains `name`, `description`, and `parameters` (a JSON-schema-like
	dict the LLM can use to craft valid calls.
	"""
	specs = {}
	
	# Add core tool specs
	for name, spec in _CORE_TOOL_SPECS.items():
		specs[name] = {"description": spec.description, "parameters": spec.parameters}
	
	# Add modular tool specs (from YouTube, Spotify, etc.)
	for tool_def in youtube_tools:
		specs[tool_def.name] = {
			"description": tool_def.spec.description,
			"parameters": tool_def.spec.parameters,
		}
	
	return specs


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
	"""Single entrypoint to execute a named tool with args (dict).

	Designed to be called by an LLM runner after validating `args` against the
	tool's `parameters` schema.
	"""
	fn = _TOOLS.get(name)
	if fn is None:
		return _result_error(f"unknown tool: {name}")
	try:
		return fn(_controller, args or {})
	except Exception as e:
		return _result_error(e)


if __name__ == "__main__":
	print("=== Available Tools ===\n")
	print(json.dumps(get_tool_specs(), indent=2))
	print("\n=== Demo Execution ===\n")

	# Example: start a headed browser
	seq = [
		("start_browser", {"headless": False}),
		("yt_watch", {"query": "how to cook filet mignon"}),
		("yt_pause_play", {})
		#("start_browser", {"headless": False}),
		#("search_youtube", {"query": "Bob Ross"}),
		#("close_browser", {}),
	]

	for name, args in seq:
		print(f"Executing: {name}")
		result = execute_tool(name, args)
		print(f"Result: {json.dumps(result, indent=2)}\n")

	input()


