from dotenv import load_dotenv
import os

load_dotenv()

ENV = os.environ.get("ENV")
MODEL_NAME = "Qwen/Qwen3-0.6B"