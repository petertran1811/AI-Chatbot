#!/bin/bash

libs=("opentelemetry-api" "opentelemetry-sdk" "opentelemetry-exporter-otlp")

for lib in "${libs[@]}"; do
    if pip show "$lib" > /dev/null 2>&1; then
        pip uninstall -y "$lib" --quiet --disable-pip-version-check --root-user-action=ignore
        echo "Uninstalled: $lib"
    else
        echo "Not installed: $lib"
    fi
done

echo "Assigning CUDA_VISIBLE_DEVICES=0"
export CUDA_VISIBLE_DEVICES=0

# Change to the src directory
cd src || { echo "Failed to change directory to src"; exit 1; }

vllm serve ../Qwen3-8B --served-model-name Qwen/Qwen3-8B --port 8000 --host 0.0.0.0 --api-key token-abc123 --dtype float16 --max-model-len 10240 --enable-prefix-caching --enable-chunked-prefill --disable-log-requests --mm-processor-kwargs '{"min_pixels": 200704, "max_pixels": 802816}' --gpu-memory-utilization 0.95 &
uvicorn server:app --host 0.0.0.0 --port 8002 --log-config logging.yaml &
wait
