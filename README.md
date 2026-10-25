# Qwen Backend With Managed vLLM

The backend starts vLLM automatically through FastAPI's lifespan hook. There is
no separate model-server startup step.

## Startup Lifecycle

1. The container starts Uvicorn with one backend worker.
2. FastAPI launches a local vLLM process serving `Qwen/Qwen3-0.6B`.
3. Startup waits for vLLM's health endpoint, including the first model download.
4. The backend accepts requests on port 8000 after the model is ready.
5. Backend shutdown terminates the vLLM process group.

Startup fails if vLLM exits or does not become ready within 900 seconds. Logs from
vLLM appear in the container logs. The model cache persists in a Docker volume.

## Deployment

Use the Compose service in `compose.yaml` on the NVIDIA laptop. It requires
Docker Compose 2.30 or newer, a recent NVIDIA driver, and NVIDIA Container Toolkit
configured for GPU containers. On Windows, use Docker Desktop's WSL2 GPU support.
The macOS development machine cannot run this NVIDIA container for inference.

Starting the Compose service starts both the backend and vLLM. Keep one Uvicorn
worker and do not use reload mode: each backend process would launch another
vLLM instance on the same port and GPU. Pin the base image to a tested version
or digest before using this beyond a sample project.

## Asking Questions

From another laptop on the same network, open
`http://<server-laptop-LAN-IP>:8000/docs` and try `POST /ask`:

```json
{
  "prompt": "Explain continuous batching in vLLM.",
  "max_tokens": 512,
  "temperature": 0.7,
  "enable_thinking": false
}
```

Responses retain vLLM's chat-completion JSON structure. `GET /health` checks the
model server. vLLM's internal port 8001 is not exposed to the network.

Allow inbound port 8000 only on your trusted private network. This sample has no
authentication; do not expose it to the internet. WSL2 may require additional
LAN networking configuration.

Compose sets the context limit to 8192 tokens and GPU utilization to 0.8. These
are starting values, not a guarantee of compatibility with a particular driver
or vLLM build. Adjust `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`, and
`VLLM_STARTUP_TIMEOUT` in the service environment as needed. Keep request prompt
and output tokens within the configured context limit.