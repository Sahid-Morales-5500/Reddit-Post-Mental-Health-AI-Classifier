import os
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')


def predict_text(title: str, selftext: str = "") -> dict:

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

    formatted_text = f"Title: {title} | Text: {selftext}"

    model = joblib.load(MODEL_PATH)

    prediction = model.predict([formatted_text])[0]
    probabilities = model.predict_proba([formatted_text])[0]
    confidence = max(probabilities) * 100

    return {
        "prediction": prediction,
        "confidence": f"{confidence:.2f}%"
    }

