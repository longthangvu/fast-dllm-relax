"""Shared helpers for the inspection notebooks in docs/notebooks/.

Every notebook starts with:

    from _helpers import *
    bootstrap()

`bootstrap()` must run before anything imports tvm: it puts the in-tree TVM build on sys.path,
sets TVM_LIBRARY_PATH and nvcc's directory, and changes the working directory to the repo root
(artifact paths in the notebooks are repo-relative). Nothing in this module imports tvm at import
time for the same reason.

Sections:
  environment      bootstrap, env_report, rel, show_source
  tiny model       tiny_config, export_model, random_params, tvm_params
  pipeline         make_recorder (IRRecorder), lower, build, pass_table
  IR display       ir_stats, op_counts, find_functions, show_ir, diff_ir, list_functions
  dlight           dlight_rule_table
  VM executable    vm_functions, call_summary, bytecode, load_executable
  numerics         to_numpy, compare, print_table
  real model       load_hf, load_relax, chat_ids, hf_to_relax_params, make_probe
  traced decoding  hf_traced_generate, relax_traced_generate
"""

from __future__ import annotations

import collections
import dataclasses
import difflib
import functools
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent
DUMP_DIR = NOTEBOOK_DIR / "_dumps"
CUDA_BIN = Path("/usr/local/cuda-12.8/bin")

__all__ = [
    "REPO_ROOT", "DUMP_DIR",
    "bootstrap", "env_report", "rel", "show_source",
    "tiny_config", "export_model", "random_params", "tvm_params",
    "make_recorder", "lower", "build", "pass_table",
    "ir_stats", "op_counts", "find_functions", "show_ir", "diff_ir", "list_functions",
    "dlight_rule_table",
    "vm_functions", "call_summary", "bytecode", "classify_callee", "load_executable",
    "to_numpy", "compare", "print_table",
    "load_hf", "load_relax", "chat_ids", "hf_to_relax_params", "make_probe",
    "hf_traced_generate", "relax_traced_generate", "events_table",
]


# ── environment ──────────────────────────────────────────────────────────────


def bootstrap(*, quiet: bool = False) -> Path:
    """Make the in-tree TVM importable, put nvcc on PATH, and chdir to the repo root. Idempotent."""
    for p in (str(REPO_ROOT / "3rdparty/tvm/python"), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("TVM_LIBRARY_PATH", str(REPO_ROOT / "3rdparty/tvm/build"))
    if shutil.which("nvcc") is None and CUDA_BIN.exists():
        os.environ["PATH"] = f"{CUDA_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
    os.chdir(REPO_ROOT)

    import numpy as np
    import torch

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    torch.set_printoptions(precision=4, linewidth=120, sci_mode=False)
    import tvm  # noqa: F401  -- fail here, not deep inside a later cell

    if not quiet:
        print(f"repo root: {REPO_ROOT.name}/  (cwd set)  tvm {tvm.__version__}  torch {torch.__version__}")
    return REPO_ROOT


def rel(path: str | Path) -> str:
    """Repo-relative path string (keeps /home/... out of committed outputs)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def show_source(obj, *, max_lines: int = 40) -> None:
    """Print an object's source (with its file:line), capped."""
    import inspect

    lines, start = inspect.getsourcelines(obj)
    print(f"# {rel(inspect.getsourcefile(obj))}:{start}")
    print(_cap([ln.rstrip("\n") for ln in lines], max_lines))


def env_report() -> dict[str, Any]:
    """TVM build flags, device, and tool versions this notebook run used."""
    import subprocess

    import torch
    import transformers
    import tvm

    info = tvm.support.libinfo()
    dev = tvm.cuda(0)
    nvcc = shutil.which("nvcc")
    nvcc_ver = None
    if nvcc:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
        nvcc_ver = out.strip().splitlines()[-1] if out else None
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
    return {
        "repo commit": git.stdout.strip() or None,
        "tvm version": tvm.__version__,
        "tvm commit": (info.get("GIT_COMMIT_HASH") or "")[:9],
        "USE_CUDA": info.get("USE_CUDA"),
        "USE_CUBLAS": info.get("USE_CUBLAS"),
        "USE_CUTLASS": info.get("USE_CUTLASS"),
        "USE_METAL": info.get("USE_METAL"),
        "metal codegen": tvm.get_global_func("target.build.metal", allow_missing=True) is not None,
        "cuda device": dev.exist and torch.cuda.get_device_name(0),
        "compute capability": dev.exist and dev.compute_version,
        "nvcc": nvcc_ver,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "aarch64 cross g++": shutil.which("aarch64-linux-gnu-g++") is not None,
    }


# ── tiny model ───────────────────────────────────────────────────────────────


def tiny_config(tied: bool = False):
    """Same tiny config as tests/test_tiny_compile.py: 2 layers, readable IR, compiles in seconds.

    head_dim stays 128 because the CUDA attention callback hard-codes softmax scale 1/sqrt(128).
    """
    from fast_dllm_relax.model.cuda import FastDLLMRelaxConfig

    return FastDLLMRelaxConfig(
        vocab_size=64, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=128, rms_norm_eps=1e-6,
        rope_theta=1e6, context_window_size=256, tie_word_embeddings=tied,
        bd_size=32, pad_token_id=0,
    )


def export_model(cfg=None, *, dtype: str = "float16", cls=None):
    """Build the production nn.Module (fused QKV / gate_up) and export it to a Relax IRModule.

    Returns (model, mod, named_params).
    """
    if cls is None:
        from fast_dllm_relax.model.cuda import FastDLLMRelaxForCausalLM as cls
    cfg = cfg or tiny_config()
    model = cls(cfg, dtype=dtype, fused_qkv=True, fused_gate_up=True)
    mod, named = model.export_tvm(spec=model.get_default_spec())
    return model, mod, named


def random_params(named, seed: int = 0, dtype: str = "float16") -> dict[str, Any]:
    """Random weights by name (norm weights = 1 to keep activations in a sane range)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = {}
    for name, p in named:
        shape = [int(d) for d in p.shape]
        if name.endswith("norm.weight"):
            out[name] = np.ones(shape, dtype=dtype)
        else:
            out[name] = (0.02 * rng.standard_normal(shape)).astype(dtype)
    return out


def tvm_params(named, values: dict[str, Any], dev) -> list[Any]:
    """Packed-param list in export order (what every exported function takes as its last arg)."""
    import tvm

    return [tvm.runtime.tensor(values[name], device=dev) for name, _ in named]


# ── pipeline instrumentation ─────────────────────────────────────────────────


@dataclasses.dataclass
class PassRecord:
    index: int
    name: str
    depth: int  # 0 = a pass called directly by the pipeline; >0 = nested inside another pass
    seconds: float
    stats: dict[str, Any]
    mod: Any = None  # IRModule after the pass (None when not kept)


# Wrapper passes whose "after" IR is the same as their last child's; not recorded.
_WRAPPER_PASSES = ("pipeline", "sequential")


@functools.cache
def _recorder_cls():
    from tvm.ir.instrument import pass_instrument

    @pass_instrument
    class IRRecorder:
        """Records the IRModule (and summary stats) after every pass that runs under it.

        Use via `lower(mod, target, pipeline, recorder=rec)`. `keep_modules=False` keeps only stats
        (for very large modules). `dump=True` also writes each module's TVMScript to
        docs/notebooks/_dumps/<run_name>/NN_<pass>.py (gitignored) for reading in full.
        """

        def __init__(self, *, keep_modules: bool = True, dump: bool = False, run_name: str = "run"):
            self.keep_modules = keep_modules
            self.dump_dir = DUMP_DIR / run_name if dump else None
            self.records: list[PassRecord] = []
            self._stack: list[tuple[str, float]] = []

        def run_before_pass(self, mod, info):
            self._stack.append((info.name, time.perf_counter()))

        def run_after_pass(self, mod, info):
            _name, t0 = self._stack.pop()
            seconds = time.perf_counter() - t0
            if info.name in _WRAPPER_PASSES:
                return
            rec = PassRecord(
                index=len(self.records),
                name=info.name,
                depth=sum(1 for n, _ in self._stack if n not in _WRAPPER_PASSES),
                seconds=seconds,
                stats=ir_stats(mod),
                mod=mod if self.keep_modules else None,
            )
            self.records.append(rec)
            if self.dump_dir is not None:
                self.dump_dir.mkdir(parents=True, exist_ok=True)
                (self.dump_dir / f"{rec.index:02d}_{rec.name}.py").write_text(mod.script(), encoding="utf-8")

        # ---- lookup helpers ----
        def names(self) -> list[str]:
            return [r.name for r in self.records]

        def find(self, name: str, occurrence: int = 0) -> PassRecord:
            hits = [r for r in self.records if r.name == name]
            if len(hits) <= occurrence:
                raise KeyError(f"{name!r} occurrence {occurrence} not recorded; have {self.names()}")
            return hits[occurrence]

        def after(self, name: str, occurrence: int = 0):
            """IRModule right after pass `name`."""
            return self.find(name, occurrence).mod

        def before(self, name: str, occurrence: int = 0, initial=None):
            """IRModule right before pass `name` (`initial` = the input module, for the first pass)."""
            rec = self.find(name, occurrence)
            if rec.index == 0:
                return initial
            return self.records[rec.index - 1].mod

    return IRRecorder


def make_recorder(**kwargs):
    """New IRRecorder instance (a tvm PassInstrument). See `_recorder_cls` for options."""
    return _recorder_cls()(**kwargs)


def lower(mod, target, pipeline, *, recorder=None, config: dict | None = None):
    """Run a Relax pipeline (what tvm.compile does before codegen) and return the lowered IRModule.

    Running the pipeline here rather than inside tvm.compile keeps the recorder out of the hundreds
    of TIR lowering passes that tvm.compile runs afterwards. `config` is PassContext config, e.g.
    {"relax.backend.use_cuda_graph": True}.
    """
    import tvm

    instruments = [recorder] if recorder is not None else []
    with tvm.transform.PassContext(instruments=instruments, config=config or {}), target:
        return pipeline(mod)


def build(lowered, target, *, tir_pipeline="default"):
    """Codegen + link an already-lowered module into a VM executable (no Relax passes re-run)."""
    import tvm

    return tvm.compile(lowered, target, relax_pipeline=None, tir_pipeline=tir_pipeline)


def pass_table(recorder, *, max_depth: int | None = None,
               cols=("relax_fns", "tir_fns", "codegen_fns", "calls", "call_tir", "extern_calls")):
    """One row per recorded pass: time and how the module's shape changed.

    max_depth=0 shows only the passes the pipeline calls directly (nested sub-passes hidden).
    """
    rows = []
    for r in recorder.records:
        if max_depth is not None and r.depth > max_depth:
            continue
        row = {"#": r.index, "pass": "  " * r.depth + r.name, "ms": round(r.seconds * 1e3, 1)}
        for c in cols:
            row[c] = r.stats.get(c)
        rows.append(row)
    print_table(rows)


# ── IR display ───────────────────────────────────────────────────────────────


def ir_stats(mod) -> dict[str, Any]:
    """Counts that summarize an IRModule: functions by kind, and calls by op/callee in Relax functions."""
    import tvm
    from tvm import relax as rx
    from tvm import tirx

    relax_fns = tir_fns = codegen_fns = composite_fns = 0
    ops: collections.Counter = collections.Counter()
    externs: collections.Counter = collections.Counter()

    def visit(e):
        if not isinstance(e, rx.Call):
            return
        op = e.op
        if isinstance(op, tvm.ir.Op):
            ops[op.name] += 1
            if e.args and isinstance(e.args[0], rx.ExternFunc):
                externs[str(e.args[0].global_symbol)] += 1
        elif isinstance(op, rx.ExternFunc):
            ops["<extern call>"] += 1
            externs[str(op.global_symbol)] += 1
        elif isinstance(op, rx.GlobalVar):
            ops["<global call>"] += 1
        else:
            ops["<local call>"] += 1

    for _gv, func in mod.functions_items():
        if isinstance(func, tirx.PrimFunc):
            tir_fns += 1
            continue
        if isinstance(func, rx.Function):
            relax_fns += 1
            attrs = func.attrs
            if attrs is not None and "Codegen" in attrs:
                codegen_fns += 1
            if attrs is not None and "Composite" in attrs:
                composite_fns += 1
            rx.analysis.post_order_visit(func.body, visit)

    return {
        "relax_fns": relax_fns,
        "tir_fns": tir_fns,
        "codegen_fns": codegen_fns,
        "composite_fns": composite_fns,
        "calls": sum(ops.values()),
        "call_tir": ops.get("relax.call_tir", 0),
        "extern_calls": sum(externs.values()),
        "ops": ops,
        "externs": externs,
        "external_mods": len((mod.attrs or {}).get("external_mods", [])) if mod.attrs is not None else 0,
    }


def op_counts(mod, fn: str) -> collections.Counter:
    """Calls inside one Relax function, by op name (extern callees as `extern:<symbol>`)."""
    import tvm
    from tvm import relax as rx

    counts: collections.Counter = collections.Counter()

    def visit(e):
        if not isinstance(e, rx.Call):
            return
        if isinstance(e.op, tvm.ir.Op):
            counts[e.op.name] += 1
            if e.args and isinstance(e.args[0], rx.ExternFunc):
                counts[f"extern:{e.args[0].global_symbol}"] += 1
        elif isinstance(e.op, rx.ExternFunc):
            counts[f"extern:{e.op.global_symbol}"] += 1
        elif isinstance(e.op, rx.GlobalVar):
            counts[f"global:{e.op.name_hint}"] += 1

    rx.analysis.post_order_visit(mod[fn].body, visit)
    return counts


def find_functions(mod, pattern: str = "", *, kind: str | None = None, codegen: str | None = None) -> list[str]:
    """Names of globals matching a regex; kind="tir"/"relax" filters by function type, codegen="cublas"/
    "cutlass" keeps only BYOC functions for that backend."""
    from tvm import relax as rx
    from tvm import tirx

    out = []
    for gv, func in mod.functions_items():
        name = gv.name_hint
        if pattern and not re.search(pattern, name):
            continue
        if kind == "tir" and not isinstance(func, tirx.PrimFunc):
            continue
        if kind == "relax" and not isinstance(func, rx.Function):
            continue
        if codegen is not None:
            attrs = func.attrs
            if not (isinstance(func, rx.Function) and attrs is not None and "Codegen" in attrs
                    and str(attrs["Codegen"]) == codegen):
                continue
        out.append(name)
    return sorted(out)


def list_functions(mod, *, pattern: str | None = None) -> None:
    """Every global in the module with its kind (Relax / PrimFunc / BYOC) and size in script lines."""
    from tvm import relax as rx
    from tvm import tirx

    rows = []
    for gv, func in sorted(mod.functions_items(), key=lambda kv: kv[0].name_hint):
        name = gv.name_hint
        if pattern and not re.search(pattern, name):
            continue
        if isinstance(func, tirx.PrimFunc):
            kind = "PrimFunc" + (" (scheduled)" if func.attrs is not None and "tirx.is_scheduled" in func.attrs else "")
        elif isinstance(func, rx.Function):
            attrs = func.attrs
            if attrs is not None and "Codegen" in attrs:
                kind = f"Relax -> BYOC {attrs['Codegen']}"
            elif attrs is not None and "Composite" in attrs:
                kind = f"Relax composite {attrs['Composite']}"
            elif attrs is not None and "global_symbol" in attrs or name in _ENTRY_POINTS:
                kind = "Relax (entry)"
            else:
                kind = "Relax"
        else:
            kind = type(func).__name__
        rows.append({"function": name, "kind": kind, "lines": func.script().count("\n") + 1})
    print_table(rows)


_ENTRY_POINTS = {
    "embed", "forward_hidden", "get_logits", "build_prefix_cache", "forward_cached", "forward_cached_block",
    "forward_block_partial_rp0", "forward_block_partial_rp8", "forward_block_partial_rp16",
    "forward_block_partial_rp24",
}


def _script(obj, fn: str | None) -> str:
    target = obj[fn] if fn is not None else obj
    return target.script()


def _cap(lines: list[str], max_lines: int) -> str:
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"])


def show_ir(obj, fn: str | None = None, *, max_lines: int = 60, grep: str | None = None, context: int = 0) -> None:
    """Print TVMScript for a module or one function, capped at `max_lines`.

    grep: only lines matching this regex (plus `context` lines around each match).
    """
    lines = _script(obj, fn).splitlines()
    if grep:
        rx_ = re.compile(grep)
        keep: set[int] = set()
        for i, line in enumerate(lines):
            if rx_.search(line):
                keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))
        picked, last = [], -2
        for i in sorted(keep):
            if i != last + 1 and picked:
                picked.append("    ...")
            picked.append(lines[i])
            last = i
        lines = picked
    print(_cap(lines, max_lines))


def diff_ir(before, after, fn: str | None = None, *, max_lines: int = 80, context: int = 2) -> None:
    """Unified diff of two modules (or of one function in both), capped."""
    a = _script(before, fn).splitlines() if before is not None else []
    b = _script(after, fn).splitlines() if after is not None else []
    diff = list(difflib.unified_diff(a, b, "before", "after", n=context, lineterm=""))
    print(_cap(diff, max_lines) if diff else "(no change)")


# ── dlight ───────────────────────────────────────────────────────────────────


def dlight_rule_table(mod, target, rules, *, show: bool = True) -> dict[str, str]:
    """Which dlight rule ApplyDefaultSchedule would pick for each unscheduled PrimFunc.

    Mirrors dlight's own loop (first rule whose apply() returns a schedule wins), so pass the same
    rule list, in the same order, as the pipeline. Run on the module *before* ApplyDefaultSchedule.
    """
    from tvm import tirx

    picked: dict[str, str] = {}
    with target:
        for gv, func in mod.functions_items():
            if not isinstance(func, tirx.PrimFunc):
                continue
            if func.attrs is not None and "tirx.is_scheduled" in func.attrs:
                picked[gv.name_hint] = "(already scheduled)"
                continue
            choice = "(none)"
            for rule in rules:
                if rule.apply(func, target, False) is not None:
                    choice = type(rule).__name__
                    break
            picked[gv.name_hint] = choice
    if show:
        counts = collections.Counter(picked.values())
        print_table([{"rule": k, "prim_funcs": v} for k, v in counts.most_common()])
    return picked


# ── VM executable ────────────────────────────────────────────────────────────


def vm_functions(ex) -> dict[str, list[str]]:
    """Parse `ex.as_text()` into {function: [instruction lines]} (packed-func stubs dropped)."""
    out: dict[str, list[str]] = {}
    current = None
    for line in ex.as_text().splitlines():
        if line.startswith("@") and line.rstrip().endswith(":"):
            current = line[1:].rstrip()[:-1]
            out[current] = []
        elif line.startswith("@"):
            current = None
        elif current is not None and line.strip():
            out[current].append(line.rstrip())
    return out


def load_executable(so_path: str | Path):
    """VMExecutable for a saved artifact library (for as_text()/stats(); load for running via load_relax)."""
    import tvm

    return tvm.relax.VMExecutable(tvm.runtime.load_module(str(so_path)))


def classify_callee(name: str) -> str:
    if "cublas" in name:
        return "cuBLAS"
    if "cutlass" in name:
        return "CUTLASS"
    if name.startswith("fast_dllm."):
        return "attention callback"
    if name.startswith("vm.builtin.") and ("alloc" in name or "null_value" in name or "kill" in name):
        return "memory builtin"
    if name.startswith("vm.builtin.") or name.startswith("vm."):
        return "other builtin"
    return "TIR kernel"


def call_summary(ex, fn: str) -> collections.Counter:
    """Calls one VM function makes, grouped by callee kind (cuBLAS / CUTLASS / callback / TIR / builtin)."""
    counts: collections.Counter = collections.Counter()
    for line in vm_functions(ex)[fn]:
        parts = line.split()
        if parts and parts[0] == "call":
            counts[classify_callee(parts[1])] += 1
    return counts


def bytecode(ex, fn: str, *, max_lines: int = 40, grep: str | None = None) -> None:
    lines = vm_functions(ex)[fn]
    if grep:
        lines = [ln for ln in lines if re.search(grep, ln)]
    print(_cap(lines, max_lines))


# ── numerics ─────────────────────────────────────────────────────────────────


def to_numpy(x):
    """float32 numpy from a torch tensor, TVM tensor (incl. bf16) or numpy array."""
    import numpy as np

    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        try:
            import torch

            return torch.from_dlpack(x).float().cpu().numpy()
        except Exception:  # noqa: BLE001 -- non-DLPack-able: go through host numpy
            x = x.numpy()
    return np.asarray(x).astype("float32")


def compare(a, b, *, name: str = "", argmax_axis: int | None = None) -> dict[str, Any]:
    """Summary of a (candidate) vs b (reference): max/mean abs error, bitwise-equal fraction, argmax agreement."""
    import numpy as np

    a = to_numpy(a).astype("float64")
    b = to_numpy(b).astype("float64")
    if a.shape != b.shape:
        raise ValueError(f"{name}: shape {a.shape} vs {b.shape}")
    d = np.abs(a - b)
    row = {
        "name": name,
        "shape": "x".join(map(str, a.shape)),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel": float(d.max() / max(np.abs(b).max(), 1e-30)),
        "equal_frac": float((a == b).mean()),
    }
    if argmax_axis is not None:
        row["argmax_agree"] = float((a.argmax(argmax_axis) == b.argmax(argmax_axis)).mean())
    return row


def print_table(rows: list[dict[str, Any]], *, floatfmt: str = ".4g") -> None:
    """Plain aligned table (no pandas dependency)."""
    if not rows:
        print("(empty)")
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)

    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return "" if v is None else str(v)

    cells = [[fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


# ── real model ───────────────────────────────────────────────────────────────


def load_hf(*, dtype="bfloat16", device: str = "cuda:0"):
    """(tokenizer, model) for the pinned Fast-dLLM v2 1.5B checkpoint (HF eager, trust_remote_code)."""
    import torch

    from fast_dllm_relax.common import DEFAULT_MODEL_NAME
    from fast_dllm_relax.hf_generation import load_fast_dllm

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    return load_fast_dllm(DEFAULT_MODEL_NAME, device=device, torch_dtype=torch_dtype)


def load_relax(artifact_dir: str = "artifacts/fast_dllm_relax_bf16", **kwargs):
    """A loaded FastDLLMRelaxRunner (exactly the runner's load path): .vm, .params, .dev, .reshape, .tokenizer."""
    from fast_dllm_relax.runners import FastDLLMRelaxRunner

    return FastDLLMRelaxRunner(artifact_dir=artifact_dir, load_model=True, **kwargs)


def chat_ids(tokenizer, prompt: str) -> list[int]:
    """Prompt ids exactly as both runners build them (chat template + generation prompt)."""
    return list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
    ))


def hf_to_relax_params(named, hf_module, dev, *, prefix: str = "") -> list[Any]:
    """Packed params for a Relax module whose attribute tree mirrors `hf_module` (zero-copy on CUDA).

    Handles the two fusions the Relax model makes: q/k/v_proj -> qkv_proj and gate/up_proj ->
    gate_up_proj (concatenated along the output dim, the same order as export_utils.remap_params).
    `prefix` is stripped from Relax names first (e.g. "inner." for the probe wrappers).
    """
    import torch
    import tvm

    sd = hf_module.state_dict()
    out = []
    for name, _ in named:
        local = name[len(prefix):] if prefix and name.startswith(prefix) else name
        if ".qkv_proj." in local or local.startswith("qkv_proj."):
            parts = [sd[local.replace("qkv_proj", p)] for p in ("q_proj", "k_proj", "v_proj")]
            t = torch.cat(parts, dim=0)
        elif ".gate_up_proj." in local or local.startswith("gate_up_proj."):
            parts = [sd[local.replace("gate_up_proj", p)] for p in ("gate_proj", "up_proj")]
            t = torch.cat(parts, dim=0)
        else:
            t = sd[local]
        out.append(tvm.runtime.from_dlpack(t.detach().contiguous().to(f"cuda:{dev.index}")))
    return out


# Entry names the production CUDA pipeline partitions for cuBLAS (`_cublas_hot_fns` in
# model/cuda.py). FuseOpsByPattern looks each one up by name, so a module must define all of them.
_CUBLAS_HOT_FNS = (
    "forward_hidden", "get_logits", "forward_cached", "forward_cached_block",
    "forward_block_partial_rp0", "forward_block_partial_rp8", "forward_block_partial_rp16",
    "forward_block_partial_rp24", "build_prefix_cache",
)


@functools.cache
def _probe_classes():
    from tvm.relax.frontend import nn
    from tvm.relax.frontend.nn import Tensor, op

    from fast_dllm_relax.model.cuda import FastDLLMRMSNorm

    def _identity(self, x: Tensor) -> Tensor:
        return x

    class _Base(nn.Module):
        pass

    # Identity stubs for the other hot names, so the unmodified production pipeline accepts a probe.
    for _name in _CUBLAS_HOT_FNS[1:]:
        setattr(_Base, _name, _identity)

    class OneInput(_Base):
        def __init__(self, inner):
            self.inner = inner

        def forward_hidden(self, x: Tensor) -> Tensor:
            return self.inner(x)

    class TwoInput(_Base):
        def __init__(self, inner):
            self.inner = inner

        def forward_hidden(self, x: Tensor, attention_bias: Tensor) -> Tensor:
            return self.inner(x, attention_bias)

    class Logits(_Base):
        def __init__(self, cfg, dtype):
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
            self.norm = FastDLLMRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)

        def forward_hidden(self, x: Tensor) -> Tensor:
            # model.norm then the tied get_logits (model/cuda.py FastDLLMRelaxForCausalLM.get_logits)
            logits = op.matmul(self.norm(x), op.permute_dims(self.embed_tokens.weight))
            return logits.astype("float32")

    return OneInput, TwoInput, Logits


def make_probe(kind: str, inner_or_cfg, *, dtype: str, hidden_size: int, num_heads: int | None = None):
    """A module that exports ONE piece of the model as `forward_hidden`, plus its export spec.

    kind="one":    forward_hidden(x) = inner(x)                      (FastDLLMRMSNorm, FastDLLMRelaxMLP)
    kind="two":    forward_hidden(x, attention_bias) = inner(x, bias) (attention, decoder layer)
    kind="logits": forward_hidden(x) = fp32(norm(x) @ embed_tokens.weight^T); pass the config.

    Using the name `forward_hidden` (and identity stubs for the other hot names) lets the probe go
    through the unmodified production pipeline, so its GEMMs get the same cuBLAS offload as the
    real artifact. Returns (probe, spec) for `probe.export_tvm(spec=spec)`.
    """
    from tvm.relax.frontend import nn

    OneInput, TwoInput, Logits = _probe_classes()
    packed = {"param_mode": "packed", "effect_mode": "none"}
    x = nn.spec.Tensor([1, "seq_len", hidden_size], dtype)
    if kind == "one":
        probe, args = OneInput(inner_or_cfg), {"x": x}
    elif kind == "two":
        bias = nn.spec.Tensor([1, num_heads, "seq_len", "seq_len"], "float32")
        probe, args = TwoInput(inner_or_cfg), {"x": x, "attention_bias": bias}
    elif kind == "logits":
        probe, args = Logits(inner_or_cfg, dtype), {"x": x}
    else:
        raise ValueError(kind)
    raw = {"forward_hidden": {**args, "$": dict(packed)}}
    for name in _CUBLAS_HOT_FNS[1:]:
        raw[name] = {"x": nn.spec.Tensor([1], "float32"), "$": dict(packed)}
    return probe, nn.spec.ModuleSpec.from_raw(raw, probe)


# ── traced decoding ──────────────────────────────────────────────────────────


def _hf_shift(logits):
    import torch

    return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)


def hf_traced_generate(model, input_ids, *, max_new_tokens: int, block_size: int = 32,
                       small_block_size: int = 8, threshold: float = 0.9, mask_id: int = 151665,
                       stop_token: int = 151645) -> tuple[list[int], list[dict]]:
    """Re-implementation of the checkpoint's `generate()` (use_block_cache=False, temperature=0) that
    records every forward call. Mirrors modeling.py's control flow line for line, including its
    prefill rule (only when the prompt is longer than one block) and the `len % block_size == 0`
    append. Returns (all token ids, events); the ids must equal `model.generate(...)`'s.
    """
    import torch

    events: list[dict] = []
    num_blocks = max_new_tokens // block_size
    original_len = input_ids.shape[1]
    step = 0

    with torch.no_grad():
        if input_ids.shape[1] > block_size:
            n = input_ids.shape[1] // block_size * block_size
            out = model(input_ids=input_ids[:, :n], use_cache=True, update_past_key_values=True, block_size=block_size)
            past = out.past_key_values
            events.append({"step": step, "kind": "prefill", "tokens_in": n, "cache_len": past.get_seq_length()})
            step += 1
            if input_ids.shape[1] % block_size == 0:
                nxt = out.logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, nxt], dim=1)
        else:
            past = None

        for _block in range(num_blocks):
            if stop_token in input_ids[:, original_len:]:
                break
            prompt_len = input_ids.shape[1]
            pad = block_size - prompt_len % block_size
            x_t = torch.cat([input_ids, torch.full((1, pad), mask_id, device=input_ids.device, dtype=torch.long)], dim=1)
            while True:
                if stop_token in x_t[:, prompt_len:]:
                    idx = (x_t[:, prompt_len:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_len:prompt_len + idx] == mask_id).sum() == 0:
                        break
                mask_idx = x_t[:, -block_size:] == mask_id
                if mask_idx.sum() == 0:
                    out = model(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past,
                                update_past_key_values=True, block_size=block_size)
                    past = out.past_key_values
                    nxt = out.logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, nxt], dim=1)
                    events.append({"step": step, "kind": "append", "tokens_in": block_size,
                                   "cache_len": past.get_seq_length(), "new_token": int(nxt)})
                    step += 1
                    break
                for sb in range(block_size // small_block_size):
                    s0 = sb * small_block_size
                    start = -block_size + s0
                    end = None if block_size == s0 + small_block_size else -block_size + s0 + small_block_size
                    while True:
                        mask_idx = x_t[:, -block_size:] == mask_id
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_len:]:
                            idx = (x_t[:, prompt_len:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_len:prompt_len + idx] == mask_id).sum() == 0:
                                break
                        out = model(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past,
                                    update_past_key_values=False)
                        logits = _hf_shift(out.logits)[:, start:end]
                        p = torch.softmax(logits, dim=-1)
                        x1 = p.argmax(dim=-1)
                        x1_p = torch.gather(p, -1, x1[..., None])[..., 0]
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)
                        unmask = x1_p > threshold
                        unmask[0, x1_p.argmax(dim=-1)] = True
                        unmask = unmask & mask_idx[:, start:end]
                        x_t[:, start:end][unmask] = x1[unmask]
                        rows = unmask[0].nonzero().flatten().tolist()
                        events.append({
                            "step": step, "kind": "full", "sub_block": sb, "tokens_in": block_size,
                            "cache_len": past.get_seq_length() if past is not None else 0,
                            "unmasked_rows": rows,
                            "tokens": x1[0, rows].tolist(),
                            "conf": [round(float(x1_p[0, r]), 4) for r in rows],
                        })
                        step += 1
            input_ids = x_t

        ids = input_ids[0].tolist()
        tail = ids[original_len:]
        if stop_token in tail:
            ids = ids[: original_len + tail.index(stop_token) + 1]
    return ids, events


def relax_traced_generate(runner, prompt_ids: list[int], *, max_new_tokens: int,
                          threshold: float | None = None) -> tuple[list[int], list[dict]]:
    """Re-implementation of runtime.relax_greedy_generate (use_block_cache=False) that records every
    compiled call. Uses the runtime's own step functions, so each event is one real VM call path.
    Returns (continuation ids, events); the ids must equal relax_greedy_generate's.
    """
    import numpy as np

    from fast_dllm_relax import runtime as rt
    from fast_dllm_relax.common import release_relax_memory_pool

    vm, params, dev, reshape = runner.vm, runner.params, runner.dev, runner.reshape
    bs, sbs = runner.block_size, runner.small_block_size
    thr = runner.threshold if threshold is None else threshold
    mask_id, stop = runner.mask_id, runner.stop_token
    nh = int(runner.hf_config.num_attention_heads)

    generated = list(prompt_ids)
    plen0 = len(generated)
    prefix_cache = None
    events: list[dict] = []
    step = 0

    def stop_done(seq, plen):
        suffix = seq[plen:]
        if stop is None or stop not in suffix:
            return False
        return mask_id not in suffix[: suffix.index(stop)]

    def cont(tokens):
        raw = tokens[plen0: plen0 + max_new_tokens]
        first = next((i for i, t in enumerate(raw) if t == mask_id), len(raw))
        return raw[:first]

    while len(generated) - plen0 < max_new_tokens:
        plen = len(generated)
        pad = bs - plen % bs or bs
        x_t = generated + [mask_id] * pad
        a0 = len(x_t) - bs
        use_cache = a0 != 0
        if use_cache and (prefix_cache is None or prefix_cache.prefix_len != a0):
            prefix_cache = rt.build_prefix_cache_from_ids(
                vm, params, dev, reshape, x_t[:a0], num_attention_heads=nh, block_size=bs)
            release_relax_memory_pool()
            events.append({"step": step, "kind": "prefill", "fn": "build_prefix_cache",
                           "tokens_in": a0, "cache_len": prefix_cache.prefix_len})
            step += 1
        for s0 in range(0, bs, sbs):
            s1 = min(bs, s0 + sbs)
            while True:
                local = [i for i, t in enumerate(x_t[a0 + s0: a0 + s1]) if t == mask_id]
                if not local:
                    break
                if stop_done(x_t, plen):
                    return cont(x_t), events
                if use_cache:
                    ids_arr, probs, _bc = rt.score_active_block_refine(
                        vm, params, dev, reshape, x_t[a0:], prefix_cache, block_size=bs, small_block_size=sbs)
                    blk_ids, blk_p = ids_arr[s0:s1], probs[s0:s1]
                    fn = rt.static_fn_name(vm, "forward_cached_block", bs)
                    gated = None
                else:
                    _, shifted = rt.score_sequence_block(
                        vm, params, dev, reshape, x_t, num_attention_heads=nh, block_size=bs)
                    release_relax_memory_pool()
                    small = shifted[:, s0:s1, :][0]
                    blk_ids = small.argmax(axis=-1)
                    blk_p = np.empty(small.shape[0], dtype=np.float64)
                    for r in range(small.shape[0]):
                        z = small[r].astype(np.float64)
                        e = np.exp(z - z.max())
                        blk_p[r] = e[blk_ids[r]] / e.sum()
                    gated = np.full_like(blk_p, -np.inf)
                    for r in local:
                        gated[r] = blk_p[r]
                    fn = "forward_hidden"
                if gated is None:
                    rows = [r for r in local if blk_p[r] > thr] or [
                        local[int(np.argmax([blk_p[r] for r in local]))]]
                else:
                    rows = [r for r in local if gated[r] > thr] or [int(np.argmax(gated))]
                for r in rows:
                    x_t[a0 + s0 + r] = int(blk_ids[r])
                events.append({
                    "step": step, "kind": "full", "fn": fn, "sub_block": s0 // sbs, "tokens_in": len(x_t[a0:]),
                    "cache_len": prefix_cache.prefix_len if use_cache else 0,
                    "unmasked_rows": rows, "tokens": [int(blk_ids[r]) for r in rows],
                    "conf": [round(float(blk_p[r]), 4) for r in rows],
                })
                step += 1
                if len(x_t) - plen0 >= max_new_tokens:
                    break
            if len(x_t) - plen0 >= max_new_tokens:
                break
        generated = x_t
        if len(generated) - plen0 >= max_new_tokens or stop_done(generated, plen0):
            break
        if use_cache:
            _, _, nxt, prefix_cache = rt.score_active_block_cached_fast(
                vm, params, dev, reshape, generated[a0:], prefix_cache)
            release_relax_memory_pool()
            fn = rt.static_fn_name(vm, "forward_cached", bs)
        else:
            cur, _ = rt.score_sequence_block(vm, params, dev, reshape, generated, num_attention_heads=nh, block_size=bs)
            nxt = int(cur[0, -1, :].argmax())
            release_relax_memory_pool()
            fn = "forward_hidden"
        generated.append(nxt)
        events.append({"step": step, "kind": "append", "fn": fn, "tokens_in": bs,
                       "cache_len": prefix_cache.prefix_len if use_cache else 0, "new_token": nxt})
        step += 1
        if stop is not None and nxt == int(stop):
            break
    return cont(generated), events


def events_table(events: list[dict], tokenizer=None, *, max_rows: int = 40) -> None:
    """Compact table of traced decode events (optionally with decoded tokens)."""
    rows = []
    for e in events[:max_rows]:
        r = {k: e.get(k) for k in ("step", "kind", "fn", "sub_block", "tokens_in", "cache_len") if k in e}
        if "unmasked_rows" in e:
            r["rows"] = e["unmasked_rows"]
            r["conf"] = e["conf"]
            if tokenizer is not None:
                r["text"] = repr(tokenizer.decode(e["tokens"]))
        if "new_token" in e and tokenizer is not None:
            r["text"] = repr(tokenizer.decode([e["new_token"]]))
        rows.append(r)
    print_table(rows)
    if len(events) > max_rows:
        print(f"... ({len(events) - max_rows} more events)")
