import json
import re

from core.groq_client import ask_groq


_JSON_BLOCK = re.compile(r"\[.*\]", flags=re.DOTALL)


def _fallback_prompts(user_prompt: str, institution_name: str, advisor_profile: str, brand_colors: list[str]) -> list[dict]:
    colors = ", ".join(brand_colors)
    base = (
        f"Background for {institution_name}. {user_prompt}. "
        f"Colors: {colors}. No people."
    )
    return [
        {"title": f"Concept {i + 1}", "style_note": "Campaign background", "prompt": f"{base} Variation {i + 1}."}
        for i in range(5)
    ]


async def design_five_prompts(
    user_prompt: str,
    institution_name: str,
    advisor_profile: str,
    brand_colors: list[str],
    visual_plan: str,
    brand_direction: str,
) -> list[dict]:
    color_text = ", ".join(brand_colors)
    messages = [
        {
            "role": "system",
            "content": (
                "Return ONLY a JSON array with 5 objects: title, style_note, prompt. "
                "Each prompt is a background image, no people, for logo placement."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Brief: {user_prompt}\n"
                f"Institution: {institution_name}\n"
                f"Colors: {color_text}\n"
                "Generate 5 diverse background prompts. JSON only."
            ),
        },
    ]

    raw = await ask_groq(messages, temperature=0.8)
    match = _JSON_BLOCK.search(raw)
    if not match:
        return _fallback_prompts(user_prompt, institution_name, advisor_profile, brand_colors)

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _fallback_prompts(user_prompt, institution_name, advisor_profile, brand_colors)

    valid = []
    for item in data:
        if isinstance(item, dict) and all(k in item for k in ("title", "style_note", "prompt")):
            valid.append(
                {
                    "title": str(item["title"]).strip(),
                    "style_note": str(item["style_note"]).strip(),
                    "prompt": str(item["prompt"]).strip(),
                }
            )

    if len(valid) != 5:
        return _fallback_prompts(user_prompt, institution_name, advisor_profile, brand_colors)

    return valid
