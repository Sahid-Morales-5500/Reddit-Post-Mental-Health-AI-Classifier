import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ai_ready_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')


def evaluate_model():

    print("Starting Model Evaluation")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run processing.py first.")

    # Load Model & Data
    print("Loading model and test dataset")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH).dropna(subset = ['ai_text','Label'])

    X = df['ai_text']
    y = df['Label']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42 )

    # Generate Predictions
    print("Evaluating predictions on test set")
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    evaluate_model()