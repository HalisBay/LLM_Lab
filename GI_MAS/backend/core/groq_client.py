from __future__ import annotations

import base64
import json
import asyncio

import httpx

from core.config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL


async def _post_groq(payload: dict) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY bulunamadi. Lutfen .env dosyasina ekleyin.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                response = await client.post(GROQ_API_URL, headers=headers, json=payload)
                if response.status_code == 429 and attempt < 4:
                    await asyncio.sleep(1.5 * attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt < 4:
                    await asyncio.sleep(1.2 * attempt)
                    continue
                raise

    if last_error:
        raise last_error
    raise RuntimeError("Groq istegi basarisiz")


async def ask_groq(messages: list[dict], temperature: float = 0.7, model: str = GROQ_MODEL) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = await _post_groq(payload)
    return data["choices"][0]["message"]["content"]


async def ask_groq_vision_json(image_bytes: bytes, text_prompt: str, model: str) -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
    }

    data = await _post_groq(payload)
    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content}
