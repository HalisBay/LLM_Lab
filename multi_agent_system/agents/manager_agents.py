from core.groq_client import ask_groq

async def review_blog(blog_content):
    messages = [
        {"role": "system", "content": "You are a blog post editor. Review the provided blog post in Turkish. Correct any grammatical errors, remove unnecessary parts, and check punctuation. If there are any English words, replace them with their Turkish equivalents."},
        {"role": "user", "content": f"Blog post to be reviewed: {blog_content}"}
    ]

    return await ask_groq(messages)