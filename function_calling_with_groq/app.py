import os
from dotenv import load_dotenv
from groq import Groq
import json
from datetime import datetime
from zoneinfo import ZoneInfo

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

TIMEZONE_DATA = {
    "istanbul": "Europe/Istanbul",
    "paris": "Europe/Paris",
    "tokyo": "Asia/Tokyo",
    "madrid": "Europe/Madrid",
}

def get_current_time(location):
    """Verilen şehir için güncel saati döndürür"""
    
    location_lower = location.lower()
    for key, timezone in TIMEZONE_DATA.items():
        if key in location_lower:
            current_time = datetime.now(ZoneInfo(timezone)).strftime("%I:%M %p")
            return json.dumps({"location": location, "current_time": current_time})
    return json.dumps({"location": location, "current_time": "unknown"})

def process_model_response(response):
    """Modelin cevabını işleyip, gerekirse fonksiyon çağrısı yapar"""

    messages = []
    model_message = response.choices[0].message
    messages.append(model_message)

    if model_message.tool_calls:
        for call in model_message.tool_calls:
            if call.function.name == "get_current_time":
                args = json.loads(call.function.arguments)
                result = get_current_time(args["location"])
                messages.append({
                    "tool_call_id": call.id,
                    "role": "tool",
                    "name": "get_current_time",
                    "content": result,
                })
    return messages

def run_conversation():
    """Kullanıcı ile sohbeti başlatan ve model ile etkileşimi yöneten fonksiyon"""

    query = "What's the current time in madrid and sivas?"
    messages = [{"role": "user", "content": query}]

    # Fonksiyon tanımlarını model için hazırlama
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"}
                    },
                    "required": ["location"]
                },
            }
        }
    ]

    try:
        # Model yanıtını alma
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            tools=tools,
            tool_choice="auto",
        )

        # Model yanıtını işleme
        messages = process_model_response(response)

        # Final yanıtı alma
        final_response = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )

        return final_response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {e}"

print("Conversation result:", run_conversation())
