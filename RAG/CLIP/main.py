import matplotlib.pyplot as plt
from utils import load_data
from model import model, processor, device
from index import build_index
from search import search
from rag import build_context, generate_answer

data = load_data()
images = data["image"]

index, _ = build_index(model, processor, images, device)

query = "dog running on grass"

# retrieval
results = search(model, processor, query, index, images, device)

# context
context = build_context(results, data)

# LLM answer
answer = generate_answer(query, context)

print("\nFINAL ANSWER:\n")
print(answer)

# -------------------------
# VISUALIZATION PART
# -------------------------

fig, axes = plt.subplots(1, len(results), figsize=(15, 3))

for i, (idx, score) in enumerate(results):
    img = images[idx]

    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"{score:.2f}")

plt.suptitle("Top-K Retrieved Images")
plt.show()