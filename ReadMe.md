# RetailVision AI: E-Commerce Visual Cataloging

## Executive Summary
In the fast-paced e-commerce sector, manually tagging thousands of daily product images with accurate metadata is expensive, slow, and prone to human error. This project provides a scalable, automated solution using Deep Learning. 

RetailVision AI is a Convolutional Neural Network (CNN) deployed as an interactive web application. It instantly analyzes raw product images and automatically classifies them into the correct digital marketing and inventory categories, drastically reducing manual data entry.

## The Business Value
- **Marketing Optimization:** Ensures products are instantly and accurately categorized for SEO and site search filters.
- **Cost Reduction:** Automates a highly repetitive data entry task.
- **Scalability:** The model can process thousands of images in the time it takes a human to manually tag one.
*(Note: This project strictly focuses on digital cataloging and marketing metadata, distinctly separate from backend supply chain operations).*

## Technical Architecture
- **Algorithm:** Deep Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Dataset:** Fashion MNIST (70,000 grayscale images of clothing across 10 categories)
- **Interface:** Streamlit (Interactive Python web framework)
- **Deployment:** Streamlit Community Cloud

## How to Use the Dashboard
1. Open the live web application: `[Insert Your Streamlit URL Here]`
2. Upload a clear, cropped image of a clothing item (e.g., a sneaker, a shirt, a bag). *Note: The model performs best on images with light items on dark backgrounds, mirroring its training data.*
3. The AI will extract the spatial features and output the predicted category tag.
4. Review the "Category Probability Matrix" to see the model's confidence distribution across all 10 possible categories.

## Repository Structure
- `app.py`: The main Streamlit application script containing the frontend UI and image preprocessing logic.
- `vision_model.h5`: The compiled and trained Convolutional Neural Network.
- `requirements.txt`: Dependencies required for cloud deployment.
