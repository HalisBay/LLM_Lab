# Fine-Tuning with Groq

## Project Description

This project is designed to generate data suitable for the fine-tuning process of a language model using the **Groq API**. Based on user queries, the model invokes specific tools (e.g., the `get_current_time` function to find out the current time in a city), and the messages generated during this process are saved in a `.jsonl` file. This file can be used to fine-tune the language model.

---

## What is Fine-Tuning?

Fine-tuning is the process of adapting a pre-trained language model to a specific task or dataset. In this process:
1. The model is retrained with a dataset customized for a specific task.
2. The model's outputs become more accurate and contextually appropriate for the targeted task.

In this project, the data format required for fine-tuning is created along with the model's responses to user queries and tool invocations.

---

## How Does It Work?

1. **User Query**: A query, such as the name of a city, is received from the user.
2. **Model Response**: The Groq API responds to the query and invokes tools if necessary.
3. **Tool Invocation**: The model calls a function like `get_current_time` to retrieve the required information.
4. **Response Generation**: The model generates the final response using the results returned from the tool invocation.
5. **Fine-Tuning Data Logging**: The messages generated throughout the process are saved in a `.jsonl` file named `fine_tune_data.jsonl`.

### Example Workflow

Below is an example of how the process works for a city name query:

- **User Query**: "What time is it in Ankara right now?"
- **Model Response**: "I am calling a tool to find out the time in Ankara."
- **Tool Invocation**: The `get_current_time("Ankara")` function is called and returns "15:30."
- **Final Response**: "The current time in Ankara is 15:30."
- **Logged Data**:
  ```json
  {
    "prompt": "What time is it in Ankara right now?",
    "content": "The current time in Ankara is 15:30."
  }
  ```
