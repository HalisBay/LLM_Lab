# Function Calling with Groq

This project demonstrates how to use the "function calling" feature of the Groq library to dynamically handle function calls from the model and how to use it in a Python application. The project allows the model to call external functions based on user queries and provides the results to the user.

## How It Works

1. **User Query**: The `run_conversation` function receives a query from the user and sends it to the Groq model.
2. **Response from Groq**: The Groq model decides if it needs to call additional functions (for example, `get_current_time`) based on the query.
3. **Function Call**: The model invokes the `get_current_time` function, calculates the current time in the specified city, and returns it.
4. **Final Response**: The model is called again, and the final message, along with the output of the function, is printed to the user.

## Short Code Overview

- **`get_current_time(location)`**: Determines the time zone for the given city name and returns the current time in that city. If the city isn't in the database, it returns `"unknown"`.

- **`process_model_response(response)`**: Examines the `tool_calls` elements in the Groq model’s response and runs any functions the model requests, adding the needed details to the response.

- **`run_conversation()`**: Prepares the user’s query, calls the Groq model for a response, and prints the final message (including function calls) to the screen.

## Example Query and Output

**Query**:  
"What's the current time in Madrid and Sivas?"

**Output**:  
```plaintext
Conversation result: The current time in Madrid is 02:58 PM. Unfortunately, I was unable to find the current time in Sivas.
```