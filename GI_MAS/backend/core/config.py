import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


def _as_bool(value: str | None, default: bool = True) -> bool:
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


GROQ_VISION_GUARD = _as_bool(os.getenv("GROQ_VISION_GUARD", "true"), default=True)

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "flux")
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "1024"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "1024"))
IMAGE_CONCURRENCY = int(os.getenv("IMAGE_CONCURRENCY", "1"))
IMAGE_REQUEST_DELAY = float(os.getenv("IMAGE_REQUEST_DELAY", "1.5"))
