# Image Generation with AI API

This project aims to generate images based on text descriptions using the Hugging Face API. A description provided by the user is processed through Hugging Face's **Stable Diffusion** model to create an image.

## How It Works
1. The user provides a description (e.g., "a city at sunset").
2. A request is sent to the Hugging Face API.
3. If the response is successful, the image is saved as `output.png`.