from core.groq_client import ask_groq

async def plan_task(user_request):

    messages = [
        {"role": "system", "content": "You are a planner. Create a clear and detailed plan for the task based on the user's request. Make sure the plan is actionable and well-structured."},
        {"role": "user", "content": f"Task to plan: {user_request}"}
    ]

    return await ask_groq(messages)