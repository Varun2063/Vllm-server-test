FROM vllm/vllm-openai:latest

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py constants.py llm.py ./

EXPOSE 8000
STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]