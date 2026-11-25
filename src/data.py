import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi


# Dataset Info
DATA_SET_NAME = "entenam/reddit-mental-health-dataset"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_credentials():

    load_dotenv()

    if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
        print("Kaggle credentials not found")
        os.environ['KAGGLE_USERNAME'] = input("Enter your Kaggle username: ").strip()
        os.environ['KAGGLE_KEY'] = input("Enter your Kaggle key: ").strip()




def autenthication():

    get_credentials()
    os.makedirs(DATA_DIR, exist_ok= True)

    try:
        print("Authenticating")
        api = KaggleApi()
        api.authenticate()

        print(f"Downloading and extracting {DATA_SET_NAME}")

        api.dataset_download_files(DATA_SET_NAME, path = DATA_DIR, unzip= True)
        print(f"Dataset downloaded in: {os.path.abspath(DATA_DIR)}")

    except Exception as e:
        print(f"Error during download: {e}")


if __name__ == '__main__':
    autenthication()