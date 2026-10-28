# Inspection notebooks

These notebooks follow Fast-dLLM v2 (1.5B) from the eager Hugging Face model to the compiled TVM
Relax artifact the runners load. They show what the rewrite changes, what each compiler pass does,
how the compiled model runs, and where its numbers and time differ from eager PyTorch. Read them in
order; each one ends with a link to the next.

| # | Notebook | Model | GPU | Approx. time | Topic |
|---|---|---|---|---|---|
| 00 | [Setup and environment](00_setup_and_environment.ipynb) | none | optional | seconds | Build flags, versions, artifacts, glossary, readiness check |
| 01 | [The eager HF model](01_eager_hf_model.ipynb) | real 1.5B | yes | ~1 min | Module tree, shapes and dtypes, block-causal mask, one refinement step by hand, full call trace |
| 02 | [From HF to `relax.frontend.nn`](02_relax_frontend_model.ipynb) | tiny + real config | no | seconds | The rewrite, the ten-function export spec, first Relax IR, HF → Relax weight mapping |
| 03 | [Numerical parity](03_numerical_parity.ipynb) | real weights | yes | a few min | Op-by-op HF vs Relax error, RoPE formula, error growth over 28 layers, threshold decisions |
| 04 | [The pipeline, pass by pass](04_pipeline_pass_by_pass.ipynb) | tiny | yes | ~1 min | BYOC (cuBLAS, CUTLASS), LegalizeOps, fusion, dlight rules, bf16 and no-CUTLASS variants |
| 05 | [Memory planning and VM lowering](05_memory_and_vm_lowering.ipynb) | tiny + real artifact | yes | ~1 min | Destination passing, static memory plan, CUDA graphs (off by default), VM bytecode |
| 06 | [Runtime and decode loop](06_runtime_and_decode_loop.ipynb) | real artifact | yes | a few min | Loading, the attention callback, DLPack, functional caches, call-by-call trace vs HF |
| 07 | [Profiling](07_profiling.ipynb) | real artifact | yes | a few min | Time per entry point, VM profiler, cost of the callback, one step broken down vs HF |
| 08 | [Other targets](08_other_targets.ipynb) | tiny | no (lowering) | ~1 min | Metal and Jetson pipelines compared with CUDA; builds when the toolchain is present |

"Tiny" is the 2-layer config from `tests/test_tiny_compile.py`: the same structure with smaller
constants, so the IR stays readable. "Real" is the pinned checkpoint and `artifacts/fast_dllm_relax_bf16`.

## Setup

```bash
pip install -e '.[docs]'          # jupyterlab, ipykernel, nbclient, nbformat, matplotlib
jupyter lab docs/notebooks/       # or open the folder in VS Code and pick the .venv interpreter
```

TVM does not have to be on `PYTHONPATH`. Each notebook's first cell calls `bootstrap()`
(`_helpers.py`), which:
- adds `3rdparty/tvm/python` to `sys.path`;
- sets `TVM_LIBRARY_PATH`;
- puts nvcc on `PATH`;
- changes the working directory to the repo root.

The notebooks expect the TVM build from `scripts/setup_tvm.sh` and, for 01/03/06/07, the cached
checkpoint and the bf16 CUDA artifact (`scripts/export_cuda.py`).

## Rerunning

To run a notebook from a clean kernel and save its outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace docs/notebooks/04_pipeline_pass_by_pass.ipynb
```

Rerun the affected notebooks after any change to `fast_dllm_relax/model/`, the pipelines, the
runtime, or the TVM submodule. Notebook 00 records the repo and TVM commits the outputs came from.

## Conventions for committed outputs

- Keep cell output to about 80 lines. Long IR goes through `show_ir(..., max_lines=..., grep=...)`,
  `diff_ir` or `bytecode`, which truncate and say how much was cut.
- Full IR dumps go to `docs/notebooks/_dumps/` (gitignored). `make_recorder(dump=True)` writes every
  pass's module there.
- Each `.ipynb` stays under 1 MB. Plots are small; nothing large is embedded.
- Only repo-relative paths appear in outputs (`rel()`), never `/home/...`.
- Seeds and prompts are fixed, so reruns on the same machine reproduce the outputs.

## The helpers (`_helpers.py`)

| Area | Functions |
|---|---|
| Environment | `bootstrap`, `env_report`, `rel`, `show_source` |
| Tiny model | `tiny_config`, `export_model`, `random_params`, `tvm_params` |
| Pipeline | `make_recorder` (an `IRRecorder` PassInstrument), `lower`, `build`, `pass_table` |
| IR | `ir_stats`, `op_counts`, `find_functions`, `list_functions`, `show_ir`, `diff_ir` |
| Scheduling | `dlight_rule_table`: which dlight rule takes each PrimFunc |
| Executable | `vm_functions`, `call_summary`, `bytecode`, `load_executable` |
| Numerics | `to_numpy`, `compare`, `print_table` |
| Real model | `load_hf`, `load_relax`, `chat_ids`, `hf_to_relax_params`, `make_probe` |
| Decoding | `hf_traced_generate`, `relax_traced_generate`, `events_table` |

Two of these help when looking at a new pipeline change:
- `lower(mod, target, pipeline, recorder=make_recorder())` captures the IR after every pass of any
  Relax pipeline, including passes a custom `module_pass` calls internally, without changing the
  pipeline.
- `make_probe` compiles a single submodule through the unmodified production pipeline, so its
  kernels match the real artifact's.
