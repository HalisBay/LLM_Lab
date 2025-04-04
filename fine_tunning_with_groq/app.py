import os
import json
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

def save_training_example(messages, filename="fine_tune_data.jsonl"):
    """Verilen mesaj dizisini .jsonl formatında kaydeder"""
    serialized_messages = []
    for m in messages:
        if isinstance(m, dict):
            serialized_messages.append(m)
        else:
            serialized_messages.append({
                "role": getattr(m, 'role', None),
                "content": getattr(m, 'content', None),
                "name": getattr(m, 'name', None)
            })

    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps({"messages": serialized_messages}, ensure_ascii=False) + "\n")

def run_conversation(query):
    """Modelle konuşmayı başlatır ve fine-tuning örneği üretir"""

    messages = [{"role": "user", "content": query}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    # İlk olarak model ile konuşma başlar
    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        tools=tools,
        tool_choice="auto"
    )

    model_message = response.choices[0].message
    messages.append(model_message)

    # Model mesajı tool çağrısı içeriyorsa, tool çağrısını işleyin
    if model_message.tool_calls:
        for call in model_message.tool_calls:
            args = json.loads(call.function.arguments)
            result = get_current_time(args["location"])  # Tool'dan gelen veri işleniyor

            tool_result = {
                "tool_call_id": call.id,
                "role": "tool",
                "name": call.function.name,
                "content": result
            }
            messages.append(tool_result)

            # Tool sonucu sonrası assistant cevabı oluşturuluyor
            result_data = json.loads(result)
            assistant_reply = {
                "role": "assistant",
                "content": f"Şu anda {result_data['location']}’da saat {result_data['current_time']}."
            }
            messages.append(assistant_reply)

            # Fine-tuning verisi kaydediliyor
            save_training_example(messages)
            return assistant_reply["content"]

    return "Beklenmeyen bir hata oluştu."

if __name__ == "__main__":
    user_input = input("Şehir sorusu girin: ")
    output = run_conversation(user_input)
    print("\nResponse:", output)
