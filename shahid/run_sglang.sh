export SGLANG_LOG_LEVEL=DEBUG
export SGL_ENABLE_JIT_DEEPGEMM=0

python -m sglang.launch_server \
  --model Qwen/Qwen3-0.6B-FP8 \
    --attention-backend triton \
  --mem-fraction-static 0.75 \
  --schedule-policy lpm \
  --enable-deterministic-inference
  
#   --attention-backend flashinfer \
