#!/usr/bin/env bash
# Run one lm-eval benchmark against the HF baseline or a compiled Relax artifact.
#
#   BACKEND=relax TASK=gsm8k bash scripts/eval.sh
#   BACKEND=hf    TASK=mmlu  LIMIT=20 bash scripts/eval.sh
#
# TASK: gsm8k | ifeval | humaneval | humaneval_plus | mbpp | mbpp_plus | math | mmlu | gpqa
#   gsm8k ... math are generative (block-diffusion decoding, threshold 0.9);
#   mmlu and gpqa are 5-shot loglikelihood (multiple choice).
#
# Env:
#   BACKEND       hf | relax                    (required)
#   TASK          see above                     (required)
#   ARTIFACT_DIR  Relax artifact directory      (default artifacts/fast_dllm_relax_bf16)
#   DEVICE        cuda:0 | metal:0 | ...        (default cuda:0)
#   CHUNKED=1     Relax only: chunked prefix-cache prefill (recommended on Jetson)
#   LIMIT         lm-eval --limit, for smoke runs (default: full dataset)
#   RESULTS_DIR   output directory              (default results/<backend>/<task>)
#   MODEL         HF model id                   (default Efficient-Large-Model/Fast_dLLM_v2_1.5B)
#   DRYRUN=1      print the lm-eval command(s) instead of running them
#   EXTRA_ARGS    extra lm-eval arguments, word-split (e.g. EXTRA_ARGS=--log_samples)
#
# humaneval*/mbpp* EXECUTE MODEL-GENERATED PYTHON on this machine (lm-eval's code_eval with
# --confirm_run_unsafe_code); run them in a sandbox/container you are comfortable with.
# gpqa needs HF_TOKEN for an account that accepted the gated Idavidrein/gpqa dataset terms.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

: "${BACKEND:?set BACKEND=hf|relax}"
: "${TASK:?set TASK (see header)}"
model="${MODEL:-Efficient-Large-Model/Fast_dLLM_v2_1.5B}"
device="${DEVICE:-cuda:0}"
artifact_dir="${ARTIFACT_DIR:-artifacts/fast_dllm_relax_bf16}"
results_dir="${RESULTS_DIR:-results/${BACKEND}/${TASK}}"

case "$BACKEND" in
  hf)    lm_model=fast_dllm_baseline; backend_args="" ;;
  relax) lm_model=fast_dllm_relax;    backend_args="artifact_dir=${artifact_dir}," ;;
  *) echo "BACKEND must be hf or relax, got '$BACKEND'" >&2; exit 1 ;;
esac
if [ "${CHUNKED:-0}" = "1" ]; then
  [ "$BACKEND" = "relax" ] || { echo "CHUNKED=1 is Relax-only" >&2; exit 1; }
  backend_args="${backend_args}chunked_prefill=True,"
fi

gen_args="use_block_cache=True,max_new_tokens=2048,block_size=32,small_block_size=8,threshold=0.9"
ll_args="block_size=32,small_block_size=8"
common_args="already_templated=True,show_io=False"

fewshot=()
unsafe=()
include=(--include_path lm_eval_tasks)
subtasks=()
case "$TASK" in
  gsm8k)          tasks=gsm8k_local;          fewshot=(--num_fewshot 0); kind=gen ;;
  ifeval)         tasks=ifeval;               fewshot=(--num_fewshot 0); kind=gen; include=() ;;
  humaneval)      tasks=humaneval_local;      fewshot=(--num_fewshot 0); kind=gen; unsafe=(--confirm_run_unsafe_code) ;;
  humaneval_plus) tasks=humaneval_plus_local; fewshot=(--num_fewshot 0); kind=gen; unsafe=(--confirm_run_unsafe_code) ;;
  mbpp)           tasks=mbpp_local;           kind=gen; unsafe=(--confirm_run_unsafe_code) ;;
  mbpp_plus)      tasks=mbpp_plus_local;      kind=gen; unsafe=(--confirm_run_unsafe_code) ;;
  math)
    kind=gen
    subtasks=(
      minerva_math_algebra_local
      minerva_math_counting_and_prob_local
      minerva_math_geometry_local
      minerva_math_intermediate_algebra_local
      minerva_math_num_theory_local
      minerva_math_prealgebra_local
      minerva_math_precalc_local
    )
    tasks=$(IFS=,; echo "${subtasks[*]}")
    ;;
  mmlu)           tasks=mmlu;             fewshot=(--num_fewshot 5); kind=ll; include=() ;;
  gpqa)           tasks=gpqa_main_n_shot; fewshot=(--num_fewshot 5); kind=ll; include=() ;;
  *) echo "unknown TASK '$TASK'" >&2; exit 1 ;;
esac
if [ "$kind" = gen ]; then
  task_args="$gen_args"
elif [ "$BACKEND" = relax ]; then
  task_args="block_size=32"   # Relax loglikelihood scoring only uses block_size
else
  task_args="$ll_args"
fi
if [ ${#unsafe[@]} -gt 0 ]; then
  export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
  echo "WARNING: $TASK executes model-generated code on this machine (--confirm_run_unsafe_code)." >&2
fi

limit_args=()
if [ -n "${LIMIT:-}" ]; then
  limit_args=(--limit "$LIMIT")
fi
read -r -a extra_args <<< "${EXTRA_ARGS:-}"

run() {  # $1 lm-eval task list, $2 results dir
  local cmd=(
    accelerate launch -m fast_dllm_relax.lm_eval_harness
    --model "$lm_model"
    --tasks "$1" "${fewshot[@]}" --batch_size 1
    --apply_chat_template --fewshot_as_multiturn
    "${include[@]}" "${unsafe[@]}"
    --model_args "model_name=${model},device=${device},${backend_args}${task_args},${common_args},results_dir=$2"
    --output_path "$2/lm_eval_results.json"
    "${limit_args[@]}" "${extra_args[@]}"
  )
  if [ "${DRYRUN:-0}" = "1" ]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
}

# The Relax runner evaluates MATH one subtask per process (one results dir per subtask), which
# bounds per-process memory growth over the long generative run.
if [ "$TASK" = math ] && [ "$BACKEND" = relax ]; then
  for t in "${subtasks[@]}"; do
    echo "=== $BACKEND MATH subtask: $t ==="
    run "$t" "${results_dir}/$t"
  done
else
  run "$tasks" "$results_dir"
fi
