import os
import glob
import pandas as pd


# Route
DATA_DIR =  os.path.join(os.path.dirname(__file__), '..', 'data')


def load_combine_data() -> pd.DataFrame:

    csv_files = glob.glob(os.path.join(DATA_DIR, '**', '*.csv'), recursive = True)

    labeled_files = [f for f in csv_files if 'LD ' in os.path.basename(f)]

    if not labeled_files:
        raise FileNotFoundError("Archives (LD *.csv) was not found")

    print(f"Files found to match: {len(labeled_files)}")

    dfs_combined = [pd.read_csv(file) for file in labeled_files]

    return pd.concat(dfs_combined, ignore_index= True)




def process_data(df: pd.DataFrame) -> pd.DataFrame:

    df['title'] = df['title'].fillna('')
    df['selftext'] = df['selftext'].fillna('')

    df['ai_text'] = "Title: " + df['title'] + " | Text: " + df['selftext']


    final_df = df[['ai_text', 'Label', 'subreddit']].copy()

    final_df = final_df.dropna(subset=['Label'])

    final_df = final_df[final_df['ai_text'].str.strip() != 'Title:  | Text:']

    final_df['Label'] = final_df['Label'].str.strip().str.title()

    return final_df




def main():

    print(" STARTING PROCESSING ")
    df_raw = load_combine_data()

    print(" PROCESSING FORMAT FOR AI ")
    df_processed = process_data(df_raw)

    output_path = os.path.join(DATA_DIR, 'ai_ready_dataset.csv')
    df_processed.to_csv(output_path, index = False)

    print(f"Process done, {len(df_processed)} records stored in: ")
    print(f"{output_path}")




if __name__ == '__main__':
    main()