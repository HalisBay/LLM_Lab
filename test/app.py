import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

while True:
    user_input = input("Enter your prompt (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting the program.")
        break
    else:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            # ANSI escape codes for colors
            green = "\033[92m"
            white = "\033[97m"
            reset = "\033[0m"

            print(f"{green}Groq's Response:{reset} {white}{chat_completion.choices[0].message.content}{reset}\n")
        except Exception as e:
            print(f"An error occurred: {e}")