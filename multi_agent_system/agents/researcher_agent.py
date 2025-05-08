from core.groq_client import ask_groq

async def do_research(topic):

    messages = [
        {"role": "system", "content": "You are a research assistant. produce detailed information about the topic covered."},
        {"role": "user", "content": f"Subject to be researched: {topic}"},
    ]

    return await ask_groq(messages)