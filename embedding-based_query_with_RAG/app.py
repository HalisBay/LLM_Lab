import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

green = "\033[92m"
blue = "\033[94m"
reset = "\033[0m"

# Bu veritabanı dış veri kaynağını simüle ediyor
database = [
    {"name": "Deniz Manzaralı Lüks Otel", "description": "Deniz manzaralı lüks bir otel arıyorsanız, bu otel tam size göre."},
    {"name": "Ekonomik Şehir Oteli", "description": "Şehir merkezinde ekonomik bir otel arayanlar için ideal."},
    {"name": "Havalimanına Yakın Konforlu Otel", "description": "Havalimanına yakın, konforlu bir otel arayanlar için mükemmel bir seçenek."},
    {"name": "Doğa İçinde Bungalov", "description": "Doğa ile iç içe, huzurlu bir tatil için bungalovlar."},
    {"name": "Aile Dostu Tatil Köyü", "description": "Aile dostu, çocuklar için aktiviteler sunan bir tatil köyü."}
]

query = "deniz manzaralı bir otel arıyorum"

# Veritabanı ile benzerlik hesapla
descriptions = [entry["description"] for entry in database]
query_embedding = model.encode(query, normalize_embeddings=True)
description_embeddings = model.encode(descriptions, normalize_embeddings=True)

# Benzerlikleri hesapla
similarities = util.pytorch_cos_sim(query_embedding, description_embeddings)[0]
most_similar_idx = similarities.argmax()
most_similar_description = descriptions[most_similar_idx]
most_similar_name = database[most_similar_idx]["name"]

# Prompt oluşturma
prompt = f"Sorgu: '{query}'\nEn yakın metin:\n{most_similar_description}\n"

print(f"{blue}En yakın isim:{reset} {most_similar_name}")
print(f"{blue}En yakın metin:{reset} {most_similar_description}")

# Veritabanı ile benzer metin bulunup modelle birlikte yanıt verilir
try:
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )

    groq_answer = chat_completion.choices[0].message.content
    print(f"{green}Groq's Response:{reset} {groq_answer}")

except Exception as e:
    print(f"An error occurred: {e}")
