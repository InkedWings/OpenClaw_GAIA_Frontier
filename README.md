# OpenClaw GAIA Runner

This is the clean GAIA benchmark layout distilled from `openclaw_latest`.

## Files

- `scripts/manage_vllm.sh`: standalone vLLM lifecycle helper (`start`, `stop`, `status`).
- `scripts/run_gaia_concurrency.py`: GAIA concurrency runner.
- `scripts/run_tp4_concurrency.sh`: TP4 concurrency experiment wrapper.
- `scripts/submit_tp4_concurrency.sbatch`: minimal Slurm submit example.
- `config/tp4_concurrency.env`: edit this to change model, TP, concurrency, context length, rounds, etc.
- `config/openclaw.template.json`: OpenClaw template config. Worker configs are generated from this.
- `data/gaia_2023_all_validation/rows.jsonl`: full GAIA 2023 validation rows in runner format.

`.local` contains the copied Node/OpenClaw install used by the runner. The
runner calls `.local/npm/lib/node_modules/openclaw/openclaw.mjs` directly.

## Submit TP4 Concurrency Example

```bash
sbatch /lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA/scripts/submit_tp4_concurrency.sbatch
```

To run inside an existing `salloc`:

```bash
bash /lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA/scripts/run_tp4_concurrency.sh
```

## Tune Parameters

Edit:

```bash
/lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA/config/tp4_concurrency.env
```

The default TP4 sweep is:

```bash
TP_LIST=4
CONCURRENCY_LIST=1,2,4,6,8
ROUNDS=3
IDX_START=0
IDX_END=8
VLLM_CONTEXT_WINDOW=32768
OPENCLAW_MAX_OUTPUT_TOKENS=4096
OPENCLAW_CONTEXT_MARGIN=1024
```

GAIA row selection is by original dataset idx and uses a Python-style half-open
range: `[IDX_START, IDX_END)`. For example:

```bash
IDX_START=20
IDX_END=40
```

Set `IDX_END=-1` to run from `IDX_START` through the end of `ROWS_JSONL`.

`OPENCLAW_CONTEXT_WINDOW` is left empty by default and auto-computed as:

```text
VLLM_CONTEXT_WINDOW - OPENCLAW_MAX_OUTPUT_TOKENS - OPENCLAW_CONTEXT_MARGIN
```

This margin avoids vLLM rejecting prompts after chat-template/tool-rendering token overhead.

## Refresh GAIA Data

```bash
HF_TOKEN=... /lustre/orion/gen150/scratch/zye25/conda-envs/AGAI/bin/python \
  /lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA/scripts/prepare_gaia_data.py \
  --idx-start 0 \
  --idx-end -1 \
  --download-attachments \
  --output-dir /lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA/data/gaia_2023_all_validation
```

## vLLM Only

The GAIA runner starts/stops vLLM automatically. For manual backend control:

```bash
source /lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA/config/tp4_concurrency.env
export SLURM_TARGET_JOB_ID="$SLURM_JOB_ID"
export SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST"
export TP_SIZE=4
export RUN_TAG="manual_tp4_p${PORT}"
bash /lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA/scripts/manage_vllm.sh start
bash /lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA/scripts/manage_vllm.sh status
bash /lustre/orion/gen150/scratch/zye25/Agentic/openclaw_GAIA/scripts/manage_vllm.sh stop
```
