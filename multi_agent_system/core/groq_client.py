import httpx
from core.config import GROQ_API_KEY, GROQ_API_URL, Model

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

async def ask_groq(messages, model=Model, temperature=0.7):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GROQ_API_URL,
            headers=HEADERS,
            json={"model": model, "messages": messages, "temperature": temperature}
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
