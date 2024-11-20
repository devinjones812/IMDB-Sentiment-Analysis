# IMDB Sentiment Analysis Project

## Overview
This project involves sentiment analysis of the IMDB dataset, leveraging both labeled and unlabeled data to enhance model performance. It includes baseline models, semi-supervised learning, BERT fine-tuning, and interpretability features.

## Setup Instructions
1. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Download the IMDB dataset and place it in the `data/` directory.
3. Run scripts in `scripts/` or explore notebooks in `notebooks/`.
4. Start the Streamlit app:
    ```bash
    streamlit run app/app.py
    ```

## Features
- Baseline Logistic Regression model
- Semi-supervised learning with label propagation
- Fine-tuning BERT for sentiment analysis
- SHAP-based model interpretability
- Interactive Streamlit app for predictions
