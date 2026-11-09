# 🍷 Wine Quality Prediction System

This project builds a machine learning prototype to classify red wine samples into:
- **GOOD QUALITY** — original quality score ≥ 6
- **NEEDS IMPROVEMENT** — original quality score < 6

## Purpose
This is a **preliminary screening tool** to help wine producers decide which samples require full laboratory testing. It is NOT a final certification.

## Data Source
UCI Machine Learning Repository — Wine Quality Dataset
https://archive.ics.uci.edu/ml/datasets/Wine+Quality

## Technologies Used
- Python 3.10+
- Pandas — data processing
- Scikit-learn — machine learning (Random Forest Classifier)
- Streamlit — web application interface
- Joblib — model serialization

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Launch the web app
streamlit run app.py
