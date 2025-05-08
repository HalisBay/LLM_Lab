from core.groq_client import ask_groq

async def write_blog(content):
    messages = [
        {"role": "system", "content": "You are a writer. You have received content in English. Your task is to understand the content and re-write it in Turkish. You should not just translate it directly, but also interpret and summarize it in a way that makes sense for a Turkish audience."},
        {"role": "user", "content": f"Content to be written in Turkish: {content}"}
    ]
    return await ask_groq(messages)
