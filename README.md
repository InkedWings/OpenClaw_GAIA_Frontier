# OpenClaw GAIA Frontier Runner

This repository contains a Frontier-oriented GAIA benchmark harness for
OpenClaw + vLLM. It starts a single-node vLLM OpenAI-compatible backend,
runs GAIA cases through OpenClaw workers at a fixed concurrency, samples
GPU/CPU resource counters, and builds per-run reports.

The current default backend mode is the compiled vLLM path:

- `ENFORCE_EAGER=0`
- `VLLM_LOG_STATS_INTERVAL=1`
- `--enable-prefix-caching`
- `--generation-config vllm`
- deterministic generation override: temperature 0, top_p 1, repetition 1

Use `ENFORCE_EAGER=1` only when cold-start latency matters more than
steady-state decode throughput.

## Layout

- `scripts/run_tp4_concurrency.sh`: main Slurm/allocation wrapper. Despite the
  name, it is also reused by TP2/TP8 submit scripts via `CONFIG_FILE`.
- `scripts/run_gaia_concurrency.py`: Python runner that starts/stops vLLM,
  launches OpenClaw workers, samples resources, and writes raw metrics.
- `scripts/manage_vllm.sh`: single-node vLLM lifecycle helper.
- `scripts/build_gaia_concurrency_report.py`: report and aggregate builder.
- `scripts/compare_vllm_compile_tp4_cc2.sh`: SSH-only TP4/cc2 comparison of
  compiled vLLM versus `--enforce-eager`.
- `config/tp4_concurrency.env`: default TP4 experiment config.
- `config/tp2_concurrency.env`, `config/tp8_concurrency.env`: alternate TP
  configs used by matching submit scripts.
- `config/openclaw.template.json`: template used to generate per-worker
  OpenClaw configs.
- `config/local.env.example`: local-only secret template.
- `data/gaia_2023_all_validation/rows.jsonl`: GAIA validation rows in runner
  format.

`.local` contains the local Node/OpenClaw install used by the runner. The
runner calls `.local/npm/lib/node_modules/openclaw/openclaw.mjs` directly.

## Local Secrets

Create `config/local.env` from the example and put local-only secrets there:

```bash
cp config/local.env.example config/local.env
```

`config/local.env` is ignored by git. The runner sources it automatically from
`scripts/run_tp4_concurrency.sh`, so `BRAVE_API_KEY` does not need to be
exported manually before normal runs.

## Default TP4/cc2 Run

Inside an existing Frontier allocation:

```bash
cd /lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA
bash scripts/run_tp4_concurrency.sh
```

The default `config/tp4_concurrency.env` uses:

```bash
TP_LIST=4
CONCURRENCY_LIST=2
ROUNDS=1
WARMUP_COUNT=1
IDX_START=0
IDX_END=8
VLLM_CONTEXT_WINDOW=32768
OPENCLAW_CONTEXT_WINDOW=23552
OPENCLAW_MAX_OUTPUT_TOKENS=4096
ENFORCE_EAGER=0
VLLM_LOG_STATS_INTERVAL=1
```

The output goes under:

```text
runs/${RUN_NAME}_${SLURM_JOB_ID}_${timestamp}
```

Useful files in each run:

- `report.md`: human-readable summary.
- `aggregate/throughput_summary_by_config.csv`: vLLM and request throughput.
- `aggregate/resource_summary_by_config.csv`: GPU/CPU/resource summaries.
- `aggregate/all_vllm_timeseries.jsonl`: parsed vLLM stats log points.
- `configs/tp*_cc*_r*/metrics/task_metrics.jsonl`: per-task outcomes.
- `configs/tp*_cc*_r*/metrics/gpu_samples.jsonl`: per-GPU samples.
- `configs/tp*_cc*_r*/metrics/cpu_samples.jsonl`: CPU usage samples.

## Submit Scripts

Submit examples are provided for common TP/concurrency points:

```bash
sbatch scripts/submit_tp4_cc2.sbatch
sbatch scripts/submit_tp4_cc4.sbatch
sbatch scripts/submit_tp8_cc2.sbatch
```

All submit scripts eventually call `scripts/run_tp4_concurrency.sh`; they
select the appropriate config file with `CONFIG_FILE`.

## Change Experiment Settings

Edit the relevant config file, for example:

```bash
config/tp4_concurrency.env
```

GAIA row selection uses a Python-style half-open range:

```bash
IDX_START=20
IDX_END=40
```

Set `IDX_END=-1` to run through the end of `ROWS_JSONL`.

To run a larger concurrency sweep:

```bash
CONCURRENCY_LIST=1,2,4,8
ROUNDS=3
```

To switch back to fast-start eager mode for debugging:

```bash
ENFORCE_EAGER=1
```

## vLLM Backend

The main runner starts and stops vLLM automatically through
`scripts/manage_vllm.sh`. Manual control is also possible inside an allocation:

```bash
source config/tp4_concurrency.env
export SLURM_TARGET_JOB_ID="$SLURM_JOB_ID"
export SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST"
export TP_SIZE=4
export RUN_TAG="manual_tp4_p${PORT}"
bash scripts/manage_vllm.sh start
bash scripts/manage_vllm.sh status
bash scripts/manage_vllm.sh stop
```

`manage_vllm.sh` writes backend logs to:

```text
${WORK}/logs/vllm_${node}_${port}.log
```

The report parser reads vLLM lines like `Avg generation throughput: ...` and
stores them as `decode_tps` in `vllm_timeseries.jsonl`.

## SSH External Backend Mode

When Slurm commands are unavailable or slow, `run_gaia_concurrency.py` can use
an already-running backend and SSH for node-local sampling:

```bash
python scripts/run_gaia_concurrency.py \
  --job-id 0 \
  --node frontier10362 \
  --remote-runner ssh \
  --skip-slurm-validation \
  --external-base-url http://frontier10362:8011/v1 \
  --external-vllm-log /lustre/orion/gen150/scratch/zye25/Agentic/logs/vllm_frontier10362_8011.log \
  --rows-jsonl data/gaia_2023_all_validation/rows.jsonl \
  --idx-start 0 \
  --idx-end 2 \
  --tp-list 4 \
  --concurrency-list 2 \
  --rounds 1 \
  --warmup-count 0 \
  --gpu-sample-sec 1 \
  --timeout 600 \
  --max-model-len 32768 \
  --openclaw-context-window 23552 \
  --model-max-tokens 4096 \
  --disable-vllm-speculative
```

## Compile Versus Eager Comparison

The current compiled default was chosen after a TP4/cc2 comparison on
`frontier10362`. The compiled path had higher active decode throughput, while
the eager path started faster.

Re-run the comparison:

```bash
scripts/compare_vllm_compile_tp4_cc2.sh
```

Defaults:

```bash
NODE=frontier10362
PORT=8011
IDX_START=0
IDX_END=2
TP=4
CONCURRENCY=2
```

The script writes:

```text
runs/vllm_compile_compare_tp4_cc2_${timestamp}/comparison_summary.md
```

The latest observed result was:

```text
eager active decode mean:   13.02 tok/s, ready 105s
compile active decode mean: 17.35 tok/s, ready 201s
```

## Refresh GAIA Data

```bash
HF_TOKEN=... /lustre/orion/gen150/scratch/zye25/conda-envs/AGAI/bin/python \
  scripts/prepare_gaia_data.py \
  --idx-start 0 \
  --idx-end -1 \
  --download-attachments \
  --output-dir data/gaia_2023_all_validation
```
