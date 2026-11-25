from src.data import autenthication
from src.processing import main as process_data
from models.training import train_model
from models.evaluate import evaluate_model
from src.predict import predict_text
import sys

def run_pipeline():

    print("PIPELINE EXECUTION")

    print("[1/5] Checking and loading raw data...")
    autenthication()

    print("\n[2/5] Cleaning and unifying dataset...")
    process_data()

    print("\n[3/5] Training AI model...")
    train_model()

    print("\n[4/5] Evaluating performance metrics...")
    evaluate_model()

    print("\n[5/5] Testing model inference...")

    if sys.stdin.closed:
        sys.stdin = open("CON", "r")

    sample_title = input("Enter the title of the Post here: ")
    sample_text = input("Enter the text of the Post here: ")
    result = predict_text(sample_title, sample_text)

    print(f"\nSample Input Title: {sample_title}")
    print(f"Predicted Class: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")

    print("\nPIPELINE COMPLETED SUCCESSFULLY")

if __name__ == '__main__':
    run_pipeline()




