import re

import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)


def _extract_code(response: str) -> str:
    """Pull a fenced code block out of a chat-style response, if present;
    otherwise fall back to the raw response (base/completion-style output).
    Upstream's build_predictions assumes a raw continuation with no markdown
    fence, which chat/instruct models routinely add.

    Prefer the LAST fenced block that actually contains a function
    definition (not necessarily the last fence overall — a model sometimes
    follows its real answer with a second, redundant fenced block of its own
    example `assert` calls, which has no `def` at all); fall back to the
    last fenced block if none define a function; fall back to the raw
    response if there's no fence at all."""
    matches = _FENCE_RE.findall(response)
    if not matches:
        return response
    def_matches = [m for m in matches if _DEF_RE.search(m)]
    code = def_matches[-1] if def_matches else matches[-1]
    return _strip_trailing_asserts(code)


def _strip_trailing_asserts(code: str) -> str:
    """Drop the model's own self-written top-level `assert` lines (and
    everything after), if any. HumanEval's grader appends its own test
    suite + `check(entry_point)` separately, so these are never needed —
    and they're sometimes malformed, contaminating an otherwise-correct
    function with a syntax error."""
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip() == line and line.lstrip().startswith("assert "):
            return "".join(lines[:i])
    return code


def _prompt_header(prompt: str) -> str:
    """Keep only the import/helper lines preceding the target function's own
    `def` line. Chat models often restate the full signature (and its
    docstring) inside their answer; splicing the model's own restated
    definition onto the *entire* original prompt would duplicate it, so we
    only keep what the prompt has above the function (typically `from
    typing import ...`-style imports the extracted code needs)."""
    header_lines = []
    for line in prompt.splitlines(keepends=True):
        if line.lstrip().startswith("def "):
            break
        header_lines.append(line)
    return "".join(header_lines)


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [_prompt_header(doc["prompt"]) + _extract_code(r) for r in resp]
        for resp, doc in zip(resps, docs)
    ]
