from __future__ import annotations

import json

from core.groq_client import ask_groq


async def review_generated_images(user_prompt: str, institution_name: str, image_reviews: list[dict]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior reviewer agent for campaign visuals. "
                "Return a concise Turkish review with sections: Genel Durum, Olmus Olanlar, "
                "Olmayanlar, Duzeltme Onerileri."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Kullanici istegi: {user_prompt}\n"
                f"Kurum: {institution_name}\n"
                f"Gorsel kalite metrikleri: {json.dumps(image_reviews, ensure_ascii=True)}"
            ),
        },
    ]
    return await ask_groq(messages, temperature=0.2)
