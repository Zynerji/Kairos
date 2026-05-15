#!/usr/bin/env bash
# Launch the 1.3B-Mamba pretraining run on the Blackwell VM.
#
# Prerequisites (verified by `examples/finetune_deepseek_r1.py --smoke`
# passing AFTER mamba_ssm is installed):
#   * mamba_ssm + causal_conv1d installed
#   * transformers + datasets installed
#   * 24 GB VRAM, batch size tuned for memory
#
# Expected wall time: ~3-5 days for 50K steps on RTX PRO 4000 Blackwell
# at batch=4, seq_len=2048, grad_accum=4. Adjust based on actual
# tok/sec measured after the first 100 steps.
#
# Save policy: `--keep-last 3` rotates checkpoints (each ~2.6 GB at
# bf16) to stay within the 32 GB disk budget.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SAVE_DIR="${SAVE_DIR:-./mamba_1b_ckpt}"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-3e-4}"
WARMUP="${WARMUP:-2000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-50}"

cd "${PROJECT_DIR}"

# Sanity check
python3 -c "import mamba_ssm, transformers, datasets, torch; \
    assert torch.cuda.is_available(), 'no CUDA'; \
    print('deps OK; CUDA:', torch.version.cuda)"

# Choose between the canonical 1.37B (needs 8-bit Adam on 24 GB) and the
# fp32-safe 0.74B default. Set CONFIG=1p4b to opt in.
EXTRA=""
if [[ "${CONFIG:-}" == "1p4b" ]]; then
    EXTRA="--canonical-1p4b --adam-8bit"
    echo "config: canonical 1.37B + AdamW8bit"
else
    echo "config: 0.74B + fp32 AdamW"
fi

# Launch detached
nohup bash -c "PYTHONPATH=. python3 examples/train_mamba_1b.py \
    --total-steps ${TOTAL_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --seq-len ${SEQ_LEN} \
    --grad-accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --warmup-steps ${WARMUP} \
    --save-dir ${SAVE_DIR} \
    --save-every ${SAVE_EVERY} \
    --keep-last 3 \
    --log-every ${LOG_EVERY} ${EXTRA}" \
    > /tmp/mamba_pretrain.log 2>&1 &

PID=$!
echo "launched mamba 1.3B pretraining, PID=${PID}"
echo "log: /tmp/mamba_pretrain.log"
echo "ckpt: ${SAVE_DIR}"
echo "PID ${PID}" > /tmp/mamba_pretrain.pid
