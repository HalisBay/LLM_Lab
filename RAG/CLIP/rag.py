from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = AutoModelForCausalLM.from_pretrained(model_id).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def build_context(results, data):
    context = []

    for idx, score in results:
        idx = int(idx)
        text = data["text"][idx]

        context.append(f"- {text} (score: {score:.3f})")

    return "\n".join(context)


def generate_answer(query, context):
    prompt = f"""
You are a multimodal AI assistant.

User query: {query}

Retrieved images descriptions:
{context}

Explain what the images show and answer the query.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    output = llm.generate(**inputs, max_new_tokens=150)

    return tokenizer.decode(output[0], skip_special_tokens=True)
