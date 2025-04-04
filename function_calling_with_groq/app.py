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
    location_lower = location.lower()
    for key, timezone in TIMEZONE_DATA.items():
        if key in location_lower:
            current_time = datetime.now(ZoneInfo(timezone)).strftime("%I:%M %p")
            return json.dumps({
                "location": location,
                "current_time": current_time
            })
    return json.dumps({"location": location, "current_time": "unknown"})

query = "What's the current time in paris"
try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    response = chat_completion.choices[0].message.content
    location = query.split("in")[-1].strip()
    time_info = get_current_time(location)
    if json.loads(time_info)["current_time"] == "unknown":
        print(f"Sorry, I don't have timezone data for {location}.")
    else:
        print("Function response:", time_info)
except AttributeError as e:
    print("Error:", e)
    print("Please check the Groq library documentation for the correct method.")