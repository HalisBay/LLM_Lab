import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")

if not api_key:
    raise ValueError("api_key bulunamadı. Lütfen .env dosyasını kontrol edin.")

url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"

prompt = input("Lütfen resim oluşturmak için bir açıklama girin: ")

# API isteği için gerekli başlıklar ve yüklemeler
headers = {"Authorization": f"Bearer {api_key}"}
payload = {"inputs": prompt}

response = requests.post(url, headers=headers, json=payload)

# Yanıt başarılı ise resmi kaydet
if response.status_code == 200:
    with open("output.png", "wb") as f:
        f.write(response.content)
    print("Resim başarıyla kaydedildi: output.png")
else:
    print("Bir hata oluştu:", response.text)
