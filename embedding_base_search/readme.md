# Embedding-Based Text Similarity Search and Visualization

## Description
This project calculates **similarity scores** between a user-provided query and predefined texts using the **Sentence Transformers** model. After calculating the similarity scores, the **comparison of texts** is visualized, and the most similar text is highlighted.

The project computes **embeddings** for the texts, evaluates **similarity**, and visualizes the results using the **seaborn** and **matplotlib** libraries.

## Technologies Used
- **Python** (3.x)
- **Sentence Transformers**: To compute text embeddings
- **Seaborn** and **Matplotlib**: To visualize the results
- **dotenv**: To manage environment variables

## How It Works
1. Run the project to compare a query with predefined texts.
2. **Embeddings** are calculated, and the **similarity score** for each text is evaluated.
3. The results are visualized as a **bar chart**.
4. The text with the highest similarity score is **highlighted in the chart**.

## Example Usage
```python
query = "I am looking for a vacation spot surrounded by nature"
texts = [
    "If you are looking for a luxurious hotel with a sea view, this hotel is perfect for you.",
    "Ideal for those looking for an affordable hotel in the city center.",
    "A great option for those looking for a comfortable hotel near the airport.",
    "Bungalows for a peaceful vacation surrounded by nature.",
    "A family-friendly resort offering activities for children."
]
```
When this code is executed, the similarity scores between the query and the texts are calculated, and the most similar text is visualized.

### Results
- **Similarity Scores** are visualized.
- The most similar text is highlighted with a red line in the chart.
- The chart is saved as `benzerlik_grafik.png`.