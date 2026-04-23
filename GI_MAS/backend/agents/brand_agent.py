from core.groq_client import ask_groq


async def build_brand_direction(institution_name: str, brand_colors: list[str]) -> str:
    color_text = ", ".join(brand_colors)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a brand strategist. "
                "Create concise brand style instructions that can be directly used in image prompts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Institution: {institution_name}\n"
                f"Brand colors: {color_text}\n"
                "Write guidance for color balance, tone, trustworthiness, and corporate feel."
            ),
        },
    ]
    return await ask_groq(messages, temperature=0.4)
