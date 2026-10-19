import re

import evaluate as hf_evaluate


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(references, predictions):
    # NOTE: upstream's own mbpp/utils.py wraps `predictions=[predictions]`
    # here, which is correct for the *default* (no filter_list) case where
    # lm-eval hands process_results a bare completion string per doc. Once a
    # filter_list is present (ours is), lm-eval's own filtered-results
    # plumbing already delivers `predictions` pre-shaped as `[[code_str]]`
    # (one candidate list) here — matching what code_eval expects directly.
    # Re-wrapping it (as upstream does) adds a spurious third nesting level,
    # which silently produces "name 'X' is not defined" for every candidate
    # regardless of correctness (confirmed empirically while debugging this).
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)
_TRAILING_MARKER_RE = re.compile(r"\s*\[/?[A-Z]+\]\s*\Z")


def _extract_code(response: str) -> str:
    """Pull a fenced code block out of a chat-style response, if present;
    otherwise fall back to the raw response. Unlike HumanEval, MBPP's prompt
    is a natural-language instruction (no code header to preserve), so the
    extracted code is used standalone.

    Neither "first" nor "last" fenced block is reliably the real answer:
    MBPP's 3-shot prompt sometimes causes the model to re-echo the few-shot
    examples' own answers before finally answering the real question (real
    answer last), while other times the model follows its real answer with
    a redundant fenced block of its own example `assert` calls (real answer
    first, trailing block has no function at all). Prefer the LAST fenced
    block that actually contains a function definition; fall back to the
    last fenced block overall if none define a function; fall back to the
    raw response if there's no fence at all."""
    matches = _FENCE_RE.findall(response)
    if not matches:
        code = response
    else:
        def_matches = [m for m in matches if _DEF_RE.search(m)]
        code = def_matches[-1] if def_matches else matches[-1]
    code = _strip_trailing_asserts(code)
    return _strip_trailing_marker(code)


def _strip_trailing_marker(code: str) -> str:
    """Drop a trailing turn-end marker token (e.g. "[DONE]", "[END]",
    "[BEGIN]", "[/BEGIN]") the model sometimes appends, mimicking the
    few-shot exemplars' own "[DONE]" convention with a different word or a
    closing-tag style. `until` already stops generation on the known
    variants, but this is a defensive fallback for any variant not
    enumerated there — without a markdown fence around the response, this
    text lands directly in the executed code and breaks otherwise-correct
    functions with a syntax error."""
    return _TRAILING_MARKER_RE.sub("", code)


def _strip_trailing_asserts(code: str) -> str:
    """Drop the model's own self-written top-level `assert` lines (and
    everything after), if any. MBPP always appends its own correct test
    assertions separately, so these are never needed — and they're
    sometimes malformed (observed: a genuine unterminated-string typo in a
    model's own example assert), contaminating an otherwise-correct
    function with a syntax error."""
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip() == line and line.lstrip().startswith("assert "):
            return "".join(lines[:i])
    return code


def extract_code_filter(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[_extract_code(r) for r in resp] for resp in resps]


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
