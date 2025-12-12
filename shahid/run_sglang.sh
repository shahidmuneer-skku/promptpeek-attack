export SGL_ENABLE_JIT_DEEPGEMM=0
python -m sglang.launch_server --model Qwen/Qwen3-0.6B-FP8 \
 --attention-backend flashinfer \
 --mem-fraction-static 0.9 \
 --schedule-policy lpm \
 --enable-deterministic-inference \
 --mem-fraction-static 0.8 \
 --cuda-graph-max-bs 16 \
 --disable-cuda-graph