import asyncio
from contextlib import asynccontextmanager
import os
import signal
import sys
import time

import httpx

from constants import MODEL_NAME


@asynccontextmanager
async def run_vllm():
    """Own the vLLM process for the duration of the backend lifespan."""
    port = int(os.environ.get("VLLM_PORT", "8001"))
    timeout = float(os.environ.get("VLLM_STARTUP_TIMEOUT", "900"))
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        os.environ.get("VLLM_MAX_MODEL_LEN", "8192"),
        "--gpu-memory-utilization",
        os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.8"),
        start_new_session=True,
    )
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", timeout=120, trust_env=False
        ) as client:
            deadline = time.monotonic() + timeout
            while True:
                if process.returncode is not None:
                    raise RuntimeError(
                        f"vLLM exited with code {process.returncode}; check its logs above."
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "vLLM did not become ready before the startup timeout."
                    )
                try:
                    response = await client.get("/health", timeout=2)
                    if response.is_success:
                        break
                except httpx.RequestError:
                    pass
                await asyncio.sleep(1)
            yield client
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=30)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
