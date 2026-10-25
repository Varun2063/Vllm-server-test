from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
import httpx
from pydantic import BaseModel, Field

from constants import ENV, MODEL_NAME
from llm import run_vllm


@asynccontextmanager
async def lifespan(app):
    async with run_vllm() as client:
        app.state.vllm = client
        yield


app = FastAPI(title="Sample backend for vLLM server test", lifespan=lifespan)


class Question(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    enable_thinking: bool = False


@app.post("/ask")
async def ask(question: Question, request: Request):
    try:
        response = await request.app.state.vllm.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": question.prompt}],
                "max_tokens": question.max_tokens,
                "temperature": question.temperature,
                "chat_template_kwargs": {"enable_thinking": question.enable_thinking},
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=error.response.status_code, detail=error.response.text
        ) from error
    except httpx.RequestError as error:
        raise HTTPException(status_code=503, detail="vLLM is unavailable.") from error
    return response.json()


@app.get("/")
def read_me():
    return {"message": "All is well.", "environment": ENV}


@app.get("/health")
async def get_health(request: Request):
    try:
        response = await request.app.state.vllm.get("/health", timeout=2)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="vLLM is unavailable.") from error
    return {"status": "ok", "model": MODEL_NAME}

