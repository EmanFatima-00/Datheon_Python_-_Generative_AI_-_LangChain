"""
Task 3 - Multi-Tool Autonomous AI Agent
========================================
A console agent built on Gemini function calling. It autonomously chooses
between three tools - DuckDuckGo web search, a safe Python math calculator and
a local file reader - and prints its reasoning chain:
    Thought -> Action -> Observation -> ... -> Final Answer

Usage (Windows 11, PowerShell):
    python task3.py                                   (interactive)
    python task3.py --query "What is 15% of the population of France?"
"""

import argparse
import ast
import math
import operator
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import List

try:
    from google import genai
    from google.genai import errors, types
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependency: {exc}\nRun: pip install google-genai ddgs")
    sys.exit(1)

MAX_STEPS = 8                 # safety limit on the agent loop
MAX_FILE_BYTES = 1_000_000
MAX_FILE_CHARS = 8000
MAX_OBSERVATION_CHARS = 4000  # what the model sees
DISPLAY_CHARS = 600           # what the console shows
ALLOWED_FILE_TYPES = {".txt", ".md", ".csv", ".json", ".log", ".py", ".xml", ".html"}


class GeminiError(Exception):
    """Any problem talking to the Gemini API, with a user-friendly message."""


class GeminiClient:
    MAX_CANDIDATES = 4
    EXCLUDE = ("image", "tts", "audio", "live", "embedding", "robotics",
               "computer-use", "native", "vision", "imagen", "veo")

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise GeminiError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and run in PowerShell:\n"
                '  setx GEMINI_API_KEY "your-key-here"\n'
                "then close and reopen the terminal.")
        self.client = genai.Client(api_key=api_key)
        self.models = self._discover_models()
        self.index = 0

    @property
    def model(self) -> str:
        return self.models[min(self.index, len(self.models) - 1)]

    @staticmethod
    def _explain(exc: Exception) -> str:
        code = getattr(exc, "code", None)
        if code in (400, 401, 403) and "API key" in str(exc):
            return "The Gemini API key is invalid. Check GEMINI_API_KEY."
        if code == 429:
            return "Free-tier quota or rate limit reached. Wait a while and retry."
        if code == 404:
            return "The selected Gemini model is not available."
        if code in (500, 503):
            return "Gemini service is temporarily unavailable. Try again later."
        return f"Gemini API error: {exc}"

    @staticmethod
    def _rank(name: str):
        match = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        version = float(match.group(1)) if match else 0.0
        return ("lite" in name, not ("preview" in name or "exp" in name), version)

    def _discover_models(self) -> List[str]:
        override = os.environ.get("GEMINI_MODEL", "").strip()
        if override:
            return [override]
        found = []
        try:
            for m in self.client.models.list():
                name = (m.name or "").replace("models/", "", 1)
                if "generateContent" not in (m.supported_actions or []):
                    continue
                if not name.startswith("gemini-") or "flash" not in name:
                    continue
                if any(w in name for w in self.EXCLUDE):
                    continue
                found.append(name)
        except errors.APIError as exc:
            raise GeminiError(self._explain(exc)) from exc
        if not found:
            raise GeminiError("No Gemini Flash model available. Set GEMINI_MODEL manually.")
        found.sort(key=self._rank, reverse=True)
        return found[: self.MAX_CANDIDATES]

    def generate(self, contents, config=None):
        last_exc = None
        while self.index < len(self.models):
            for attempt in range(3):
                try:
                    return self.client.models.generate_content(
                        model=self.models[self.index], contents=contents, config=config)
                except errors.APIError as exc:
                    last_exc = exc
                    code = getattr(exc, "code", None)
                    text = str(exc)
                    if code in (400, 401):
                        raise GeminiError(self._explain(exc)) from exc
                    if code in (429, 500, 503):
                        if "PerDay" in text or attempt == 2:
                            break
                        found = re.search(r"retry in ([\d.]+)s", text)
                        wait = min(float(found.group(1)) + 1 if found else 15, 60)
                        print(f"  [rate limited / busy - waiting {wait:.0f}s]")
                        time.sleep(wait)
                        continue
                    break
            self.index += 1
        raise GeminiError(self._explain(last_exc) if last_exc else "No model could be used.")


# --------------------------------------------------------------------------
# Tool 1: real-time web search (DuckDuckGo)
# --------------------------------------------------------------------------
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo for current or factual information.

    Args:
        query: The search query text.

    Returns:
        The top search results (title, URL, snippet) as text.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # older package name
        results = list(DDGS().text(query, max_results=5))
    except Exception as exc:
        return f"Search failed: {exc}"
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '')}\n  {r.get('href', '')}\n  {r.get('body', '')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Tool 2: safe math calculator (AST based - no eval of arbitrary code)
# --------------------------------------------------------------------------
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs,
          "round": round, "floor": math.floor, "ceil": math.ceil}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a math expression exactly. Use this for ALL arithmetic.

    Supports + - * / // % **, parentheses, pi, e and the functions
    sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil.

    Args:
        expression: The expression, for example "0.15 * 67750000" or "sqrt(144) + 2**5".

    Returns:
        The numeric result as text, or an error message.
    """
    try:
        value = _eval_node(ast.parse(expression.strip(), mode="eval"))
        return str(value)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        return f"Error: could not evaluate '{expression}' ({exc})"


# --------------------------------------------------------------------------
# Tool 3: local file reader (restricted to the current folder)
# --------------------------------------------------------------------------
def read_file(file_path: str) -> str:
    """Read a local text file (txt, md, csv, json, log, py, xml, html) from the current folder.

    Args:
        file_path: Relative path of the file, for example "notes.txt".

    Returns:
        The file's text (truncated if very long), or an error message.
    """
    try:
        base = Path.cwd().resolve()
        path = (base / file_path).resolve()
        if base != path and base not in path.parents:
            return "Error: access denied - files must be inside the current folder."
        if not path.is_file():
            return f"Error: file not found: {file_path}"
        if path.suffix.lower() not in ALLOWED_FILE_TYPES:
            return f"Error: unsupported file type '{path.suffix}'."
        if path.stat().st_size > MAX_FILE_BYTES:
            return "Error: file is larger than 1 MB."
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_FILE_CHARS:
            return text[:MAX_FILE_CHARS] + f"\n...[truncated, {len(text)} characters total]"
        return text or "(file is empty)"
    except Exception as exc:
        return f"Error reading file: {exc}"


TOOLS = {"web_search": web_search, "calculator": calculator, "read_file": read_file}

SYSTEM_INSTRUCTION = f"""You are an autonomous research agent. Today's date is {date.today().isoformat()}.
You have three tools: web_search (current facts), calculator (all arithmetic) and read_file (local files).
Work step by step:
1. Before EVERY tool call, write one short sentence starting with "Thought:" explaining why.
2. Never do arithmetic yourself - use the calculator.
3. Use web_search for facts that may be recent or that you are unsure about; do not guess.
4. When you have enough information, stop calling tools and write the final answer clearly and concisely."""


def short(text: str, limit: int = DISPLAY_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ..."


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------
def run_agent(gemini: GeminiClient, question: str) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[web_search, calculator, read_file],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]

    for step in range(1, MAX_STEPS + 1):
        response = gemini.generate(contents, config)
        if not response.candidates or not response.candidates[0].content:
            return "The model returned no answer (it may have been blocked). Try rephrasing."
        model_content = response.candidates[0].content
        parts = model_content.parts or []
        calls = [p.function_call for p in parts if p.function_call]
        text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False)).strip()

        if not calls:
            return text or "The model returned an empty answer."

        if text:
            thought = re.sub(r"^Thought:\s*", "", text)
            print(f"\n[Step {step}] Thought: {short(thought)}")
        contents.append(model_content)  # keep the model turn exactly as returned

        response_parts = []
        for call in calls:
            args = dict(call.args or {})
            shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(f"[Step {step}] Action: {call.name}({shown})")
            tool = TOOLS.get(call.name)
            if tool is None:
                observation = f"Error: unknown tool '{call.name}'."
            else:
                try:
                    observation = tool(**args)
                except TypeError as exc:
                    observation = f"Error: invalid arguments ({exc})"
            print(f"[Step {step}] Observation: {short(observation)}")
            response_parts.append(types.Part.from_function_response(
                name=call.name, response={"result": str(observation)[:MAX_OBSERVATION_CHARS]}))
        contents.append(types.Content(role="user", parts=response_parts))

    return f"Stopped after {MAX_STEPS} steps without a final answer. Try a simpler question."


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Multi-tool autonomous agent using Gemini.")
    parser.add_argument("--query", help="ask one question and exit")
    args = parser.parse_args()

    try:
        gemini = GeminiClient()
        print(f"Agent ready (model: {gemini.model}). Tools: web_search, calculator, read_file.")
        if args.query:
            answer = run_agent(gemini, args.query)
            print(f"\nFinal Answer: {answer}")
            return
        print("Type a question, or 'exit' to quit.")
        while True:
            question = input("\nYou> ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue
            try:
                answer = run_agent(gemini, question)
                print(f"\nFinal Answer: {answer}")
            except GeminiError as exc:
                print(f"\nError: {exc}")
    except GeminiError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()