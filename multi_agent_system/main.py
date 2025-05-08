import asyncio
from agents.planner_agent import plan_task
from agents.researcher_agent import do_research
from agents.writer_agent import write_blog
from agents.manager_agents import review_blog
async def main():
    user_request = input("What would you like to do? > ")

    print("\n🧭 Starting planning...")
    plan = await plan_task(user_request)
    print("\n📋 Plan:\n", plan)

    print("\n🔍 Starting research...")
    research = await do_research(user_request)
    print("\n📚 Research:\n", research)

    print("\n✍️ Starting writing process...")
    blog = await write_blog(research)
    print("\n📝 Blog Post:\n", blog)

    print("\n🔍 Starting blog review...")
    review_feedback = await review_blog(blog)
    print("\n📝 Review Feedback:\n", review_feedback)

    with open("blog_post.md", "w", encoding="utf-8") as f:
        f.write(blog)

if __name__ == "__main__":
    asyncio.run(main())
