# Reddit Post Mental Health - AI Classifier

[![Python](https://img.shields.io/badge/PYTHON-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/SCIKIT--LEARN-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/PANDAS-DATA-blue?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Kaggle](https://img.shields.io/badge/KAGGLE-DATASET-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset)

---

## What does this AI do and how?

### What the AI does:
This is a text classifier for Reddit posts about mental health. You give it the title and text of a post, and it predicts which of these 4 topics the post is about:

* Drug and Alcohol
* Early Life
* Personality
* Trauma and Stress

The labels and data come from a public [Kaggle dataset](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset). The dataset is downloaded automatically when you run the project.

**Note:** This project classifies posts by **topic** for research and moderation. It is **not** a diagnostic tool and does not assess anyone's mental health.

### How the AI classifies:
The program runs a 5-step pipeline:

1. **Download:** Logs in to Kaggle and downloads the dataset.
2. **Processing:** Merges the labeled CSV files and joins the title and text of each post into one field.
3. **Training:** Converts the text into numbers with **TF-IDF** (single words and pairs of words, 10,000 features) and trains a **Logistic Regression** model with 80% of the data.
4. **Evaluation:** Tests the model with the other 20% and prints the classification report and the confusion matrix.
5. **Prediction:** Asks you for a "Title" and "Text" in the console and returns the **Predicted Class** with its confidence percentage.

```
Kaggle ─► Processing ─► TF-IDF + Logistic Regression ─► Evaluation ─► Prediction
```

---

## What results does it give?

<!-- Add your real results here after running the program:
| Metric | Score |
| --- | --- |
| Accuracy | 0.XX |
| Macro F1 | 0.XX |
-->

**Example output:**
```
Enter the title of the Post here: I can't stop thinking about what happened
Enter the text of the Post here: Since the accident I wake up every night...

Predicted Class: Trauma And Stress
Confidence: 81.30%
```

---

## What is its utility?
* **Automated Moderation:** Helps forum moderators or community managers quickly tag and organize posts based on their main topic.
* **Data Analysis:** Allows researchers to process large amounts of unlabeled Reddit texts to study mental health trends.
* **Smart Routing:** Can be used as a backend service to detect the topic of a user's text and route them to the right resources or support groups.

---

## How is the project composed?
The project is modularized into directories separating the data processing, AI training, and execution files.

* **`src/` (The data processing folder):**
  * `data.py`: Handles Kaggle authentication and dataset download.
  * `processing.py`: Cleans and formats the raw CSV data for the model.
  * `predict.py`: Contains the logic to classify new text.
* **`models/` (The AI scripts folder):**
  * `training.py`: Handles text vectorization and model training.
  * `evaluate.py`: Generates the classification report and confusion matrix.
* **`main.py`:** The main orchestrator that runs the entire pipeline sequentially.
* **`data/`:** Created automatically. Stores the downloaded dataset on your computer (not uploaded to GitHub).

---

## How to download and use the program:

**1. Clone the repository:**
```bash
git clone https://github.com/Sahid-Morales-5500/Reddit-Post-Mental-Health-AI-Classifier.git
cd Reddit-Post-Mental-Health-AI-Classifier
```

**2. Install the dependencies:**
```bash
pip install pandas scikit-learn joblib kaggle python-dotenv
```

**3. Get your Kaggle API key:**
Go to [kaggle.com](https://www.kaggle.com/) → Settings → API → **Create New Token**.

**4. Create your `.env` file:**

Create a file named `.env` in the main folder and add your credentials:
```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```
**Note:** If the `.env` file is missing, the program will ask for your credentials in the console.

**5. Run the program:**
```bash
python main.py
```
The program will download the data, train and evaluate the model, and then ask you for a post **Title** and **Text** to classify.

---

## What are its current limitations?
* **Word-based model:** TF-IDF counts words, so it does not understand context, negation ("I *don't* drink anymore") or sarcasm.
* **English only:** The model was trained with English posts.
* **Fixed topics:** It can only predict the 4 topics of the dataset.
* **Not a diagnostic tool:** Predictions should never be used to make decisions about a person's mental health.
