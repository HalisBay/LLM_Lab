# GI_MAS - Groq Integrated Multi-Agent Studio

Bu klasor, Groq API kullanan bir multi-agent sistem ve ona bagli bir arayuz icerir.
Sistem, kullanicidan aldigi brief'e gore logo/marka renklerine uygun 5 farkli arka plan gorseli uretir ve `images/` klasorune kaydeder.
Bu modda danisman resmi islenmez; arka planlar sonradan kisi yerlestirmeye uygun bos kompozisyonlarla uretilir.

## Kisa Cevap: Groq bu isi tek basina yapabilir mi?

- Groq, bu projede **coklu ajan orkestrasyonu** ve **prompt uretimi** icin kullaniliyor.
- Groq API tarafi chat/completion odaklidir.
- Dogrudan text-to-image uretimi genelde harici bir gorsel servisle tamamlanir.
- Bu projede gorsel URL uretimi icin Pollinations kullanildi.

## Hangi Groq modeli?

Bu sistem icin varsayilan model:
- `llama-3.3-70b-versatile`

Daha hizli ve daha ucuz alternatif:
- `llama-3.1-8b-instant`

Modeli `.env` icinde `GROQ_MODEL` ile degistirebilirsin.

Vision denetimi icin (opsiyonel):
- `GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct`
- `GROQ_VISION_GUARD=true`

Not: Vision model burada gorsel uretmez; arka planda ekstra insan olup olmadigini denetlemek icin kullanilir.

## Proje Yapisi

```text
GI_MAS/
  images/
  uploads/
  backend/
    app.py
    workflow.py
    core/
      config.py
      groq_client.py
    agents/
      planner_agent.py
      brand_agent.py
      prompt_designer_agent.py
      critic_agent.py
  frontend/
    index.html
    style.css
    app.js
  .env.example
  requirements.txt
  start.py
  readme.md
```

## Kurulum

1. Klasore gec:

```bash
cd GI_MAS
```

2. Sanal ortam olustur (opsiyonel ama onerilir):

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

3. Paketleri kur:

```bash
.venv\\Scripts\\activate
pip install -r requirements.txt
```

4. `.env.example` dosyasini `.env` olarak kopyala ve `GROQ_API_KEY` degerini doldur.

## Calistirma

Tek komutla backend + frontend birlikte acilir:

```bash
cd GI_MAS
python start.py
```

Tarayicida ac:
- `http://127.0.0.1:8000`

## Hiz Ayari

- 5 gorsel uretimi artik paralel calisir.
- Paralellik seviyesi `.env` ile ayarlanabilir:

```env
IMAGE_CONCURRENCY=3
```

- Rate limit aliyorsan (429): `IMAGE_CONCURRENCY=2` veya `1` yap.

## Arayuz Akisi

1. Kampanya promptunu yaz.
2. Kurum adini gir.
3. Kurum logosunu yukle (opsiyonel ama onerilir).
4. Marka renklerini virgul ile gir.
5. `5 Tasarim Uret` butonuna bas.

Ekranda su bilgiler gorunur:
- Kullanilan Groq modeli
- Ajan plan ciktisi
- Marka yorumu
- Canli hiyerarsik ajan durum agaci (planner, brand_agent, prompt_designer, critic, renderer, composer)
- 5 farkli gorsel karti (gorsel + prompt metni)
- Olusan dosyalar fiziksel olarak `GI_MAS/images` altina kaydedilir

## Notlar

- Kurum logosu yuklersen sistem logoyu cikti gorseline ekler.
- Logo ve danisman gorseli icin PNG/JPG kullanman onerilir.
- Marka renkleri tum promptlara acik sekilde eklenir.
- Eger LLM JSON formatinda cevap vermezse sistem fallback promptlar uretir.
- Harici gorsel serviste 429 olursa sistem yerel arka plan fallback ile gorsel uretimine devam eder.
