# Multi-Agent System for Blog Writing

## Project Description

This project demonstrates a multi-agent system designed to automate the process of creating blog posts. The system consists of multiple agents, each responsible for a specific task, such as planning, researching, writing, and reviewing. By leveraging AI capabilities, the system ensures that the generated content is well-structured, informative, and tailored to the user's request.

---

## How It Works

1. **User Input**:
   - The user provides a request or topic for the blog post.
   
2. **Planning**:
   - The `Planner Agent` creates a detailed and actionable plan for the blog post based on the user's input.

3. **Research**:
   - The `Researcher Agent` gathers detailed information about the topic to ensure the blog is informative and accurate.

4. **Writing**:
   - The `Writer Agent` interprets the research and writes the blog post in Turkish, ensuring it is well-summarized and audience-appropriate.

5. **Reviewing**:
   - The `Manager Agent` reviews the blog post, correcting grammatical errors, improving clarity, and ensuring consistency.

6. **Output**:
   - The final blog post is saved as `blog_post.md` in the project directory.

---

## Technologies Used

- **Python**: The programming language used to implement the multi-agent system.
- **httpx**: For making asynchronous HTTP requests.
- **dotenv**: For managing environment variables.
- **Groq API**: Used by agents to interact with a language model for planning, researching, writing, and reviewing.

---
## Folder Structure

- **`agents/`**:
  - Contains the implementation of the planner, researcher, writer, and manager agents.
- **`core/`**:
  - Includes configuration files and the Groq client for API interactions.
- **`main.py`**:
  - The entry point for running the multi-agent system.
- **`blog_post.md`**:
  - The output file where the generated blog post is saved.

---
## Example Workflow

1. **Run the Application**:
   ```bash
   python main.py
   ```

2. **Provide Input**:
- When prompted, enter the topic or request for the blog post. My topic is: "The Benefits of Taking Notes and Effective Note-Taking Techniques."

3. **Generated Output**:
   - The system will generate a blog post and save it as `blog_post.md`.

4. **Example Blog Post**:
   - You can find an example blog post at [blog_post.md](./blog_post.md).

---


