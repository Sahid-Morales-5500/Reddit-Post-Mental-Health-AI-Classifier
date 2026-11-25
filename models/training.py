import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ai_ready_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')


def train_model():

    print("AI Training Begins")


    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'File not found at {DATA_PATH}, run: "Processing.py". ')

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset = ['ai_text', 'Label'])

    X = df['ai_text']
    y = df['Label']


    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
    random_state = 42)

    print(f"Training records: {len(X_train)} | Test records: {len(X_test)}")


    # Create Pipeline (Text Vectorizer + Classifier)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features = 10000, 
        stop_words = 'english', ngram_range = (1,2) )  ),

        ('classifier', LogisticRegression(max_iter = 1000, C = 1.0) )

    ])


    # Train Model
    print("Training Language model")
    pipeline.fit(X_train, y_train)


    # Preliminary Evaluation
    accuracy = pipeline.score(X_test, y_test)
    print(f"Preliminary Accuracy: {accuracy * 100:.2f}%")

    # Save exported model
    os.makedirs(MODEL_DIR, exist_ok = True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model successfully saved at:\n {MODEL_PATH}")


if __name__ == '__main__':
    train_model()
