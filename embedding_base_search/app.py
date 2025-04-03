import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# ANSI escape codes for colors
green = "\033[92m"
reset = "\033[0m"

texts = [
    "Deniz manzaralı lüks bir otel arıyorsanız, bu otel tam size göre.",
    "Şehir merkezinde ekonomik bir otel arayanlar için ideal.",
    "Havalimanına yakın, konforlu bir otel arayanlar için mükemmel bir seçenek.",
    "Doğa ile iç içe, huzurlu bir tatil için bungalovlar.",
    "Aile dostu, çocuklar için aktiviteler sunan bir tatil köyü."
]
query = "doğa ile iç içe bir tatil yeri arıyorum"

# embeddings hesaplama
text_embeddings = model.encode(texts, normalize_embeddings=True)
query_embedding = model.encode(query, normalize_embeddings=True)

# benzerlik hesaplama
similarities = util.dot_score(query_embedding, text_embeddings)[0]
most_similar_index = similarities.argmax()
most_similar_text = texts[most_similar_index]
similarity_score = similarities[most_similar_index].item()

print(f"{green} En yakın metin ve skoru: {most_similar_text} , {similarity_score}{reset}")

for i, (text, score) in enumerate(zip(texts, similarities)):
    print(f"Metin {i}: {text} - Skor: {score.item()}")


text_labels = [f"Metin {i}" for i in range(len(texts))]
scores = similarities.tolist()


plt.figure(figsize=(10, 5))
sns.barplot(x=scores, y=text_labels, palette="viridis", hue=text_labels, legend=False)

# en yüksek skorlu metni vurgula
plt.axvline(similarity_score, color='r', linestyle='--', label="En Benzer Metin")

plt.xlabel("Benzerlik Skoru")
plt.ylabel("Metinler")
plt.title(f"Sorgu: \"{query}\" için Benzerlik Skorları")
plt.legend()


output_path = os.path.join("benzerlik_grafik.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
