from typing import Any, Dict, List
import json
import re
import io
import contextlib


def to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: List[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunks).strip()
    return str(value)


def safe_json_parse(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


def extract_code(raw: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", raw, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return raw.strip()


def rule_based_safeguard(code: str) -> tuple[bool, str]:
    dangerous_patterns = [
        r"\bimport\s+os\b",
        r"\bimport\s+subprocess\b",
        r"\bimport\s+socket\b",
        r"\bfrom\s+os\b",
        r"\bopen\s*\(",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\b__import__\b",
        r"\bglobals\s*\(",
        r"\blocals\s*\(",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            return False, f"Blocked by safeguard rule: pattern '{pattern}'"
    if len(code) > 4000:
        return False, "Blocked by safeguard rule: code too large."
    return True, "Passed rule-based safety checks."


def execute_python(code: str) -> tuple[str, str]:
    """Run generated code in a constrained Python namespace."""
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    globals_dict: Dict[str, Any] = {"__builtins__": safe_builtins}
    locals_dict: Dict[str, Any] = {}
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, globals_dict, locals_dict)
        out = buffer.getvalue().strip() or "(no stdout)"
        return out, ""
    except Exception as exc:  # noqa: BLE001
        return buffer.getvalue().strip(), f"{type(exc).__name__}: {exc}"
