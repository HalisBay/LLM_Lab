import json
import re

from core.groq_client import ask_groq


_JSON_BLOCK = re.compile(r"\[.*\]", flags=re.DOTALL)


async def review_and_improve_prompts(prompts: list[dict], brand_colors: list[str]) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a quality control agent for image prompts. "
                "Return ONLY a JSON array with 5 objects and keys title, style_note, prompt. "
                "Ensure every prompt is background-only (no people), preserves clear placement area, and includes brand colors."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brand colors: {', '.join(brand_colors)}\n"
                f"Prompt list: {json.dumps(prompts, ensure_ascii=True)}"
            ),
        },
    ]
    raw = await ask_groq(messages, temperature=0.3)
    match = _JSON_BLOCK.search(raw)
    if not match:
        return prompts

    try:
        checked = json.loads(match.group(0))
    except json.JSONDecodeError:
        return prompts

    if not isinstance(checked, list) or len(checked) != 5:
        return prompts

    cleaned = []
    for item in checked:
        if isinstance(item, dict) and all(k in item for k in ("title", "style_note", "prompt")):
            cleaned.append(
                {
                    "title": str(item["title"]).strip(),
                    "style_note": str(item["style_note"]).strip(),
                    "prompt": str(item["prompt"]).strip(),
                }
            )

    return cleaned if len(cleaned) == 5 else prompts
