from datasets import load_dataset


def load_data():
    data = load_dataset("jamescalam/image-text-demo", split="train")
    return data
