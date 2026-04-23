from core.groq_client import ask_groq


async def create_visual_plan(user_prompt: str, institution_name: str, advisor_profile: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a creative director agent. "
                "Create a short, practical visual strategy for 5 campaign visuals. "
                "Each visual must be background-only, with no people, and leave suitable space "
                "for later person placement."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Institution: {institution_name}\n"
                f"Advisor profile: {advisor_profile}\n"
                f"User request: {user_prompt}\n"
                "Provide a short and practical plan."
            ),
        },
    ]
    return await ask_groq(messages, temperature=0.5)
