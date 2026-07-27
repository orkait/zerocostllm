#!/usr/bin/env bash
# Serve a local model behind an OpenAI-compatible endpoint for the `local/` provider.
#
# Defaults to Ternary Bonsai 27B at 64K context - measured on a 12GB RTX 3060 at
# ~8.7GB VRAM / ~29 tok/s, which leaves headroom for a desktop session. Every knob is
# an env override, so this is not pinned to one machine or one model.
#
#   ./scripts/local-llm.sh              # serve on :8081
#   VISION=1 ./scripts/local-llm.sh     # + vision tower (adds ~0.6GB)
#   DRAFT=1  ./scripts/local-llm.sh     # + DSpark speculative drafter (lossless 1.4-2x, ~2GB)
#   CTX=131072 ./scripts/local-llm.sh   # longer context (measured ceiling on 12GB)
#
# Then point the backend at it (backend/.env or .env.local):
#   LOCAL_LLM_BASE_URL=http://127.0.0.1:8081/v1
#   LOCAL_LLM_CTX=65536
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/mnt/storage/codespace/models/bonsai-27b}"
BIN="${BIN:-/mnt/storage/codespace/code/orkait/bonsai.llama/build/bin/llama-server}"
MODEL="${MODEL:-$MODEL_DIR/Ternary-Bonsai-27B-Q2_0.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/Ternary-Bonsai-27B-mmproj-Q8_0.gguf}"
DRAFTER="${DRAFTER:-$MODEL_DIR/Ternary-Bonsai-27B-dspark-Q4_1.gguf}"

PORT="${PORT:-8081}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-65536}"
UBATCH="${UBATCH:-128}"
VISION="${VISION:-0}"
DRAFT="${DRAFT:-0}"

for f in "$BIN" "$MODEL"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

ARGS=(
  -m "$MODEL"
  -ngl 999 -fa on
  -c "$CTX"
  # 4-bit KV cache is what makes long context fit in consumer VRAM (~4x cheaper than
  # fp16 KV); the small ubatch keeps the logits buffer bounded on large-vocab models.
  -ctk q4_0 -ctv q4_0
  -ub "$UBATCH"
  -np 1
  --host "$HOST" --port "$PORT"
)

[ "$VISION" = "1" ] && [ -e "$MMPROJ" ] && ARGS+=(--mmproj "$MMPROJ")
if [ "$DRAFT" = "1" ] && [ -e "$DRAFTER" ]; then
  # Speculative decoding is lossless: the target verifies every drafted token, so the
  # output distribution is unchanged. It costs VRAM, which competes with context.
  ARGS+=(-md "$DRAFTER" --spec-type draft-dspark --spec-draft-n-max 4 -ngld 999)
fi

echo "serving $(basename "$MODEL") on http://$HOST:$PORT/v1  (ctx=$CTX vision=$VISION draft=$DRAFT)"
exec "$BIN" "${ARGS[@]}"
