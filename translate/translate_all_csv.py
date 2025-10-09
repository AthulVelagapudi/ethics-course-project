import os
from dotenv import load_dotenv
import requests
import pandas as pd
import time
from tqdm import tqdm

# ----------------- CONFIG -----------------
INPUT_DIRECTORY = "data/india/2022"
INPUT_FILE_NAME = "2022_india_persona_groups_cleaned.csv"
OUTPUT_DIRECTORY = "data/translated_questions"
LANGUAGES = ["bn", "te", "hi"]

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
URL = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"

# ----------------- TRANSLATION FUNCTIONS -----------------
def translate_text(text, target_lang, retries=3):
    """Translate a single text string using Google Translate API with retries"""
    if pd.isna(text) or str(text).strip() == "":
        return text 
    data = {
        "q": str(text),
        "target": target_lang,
        "format": "text"
    }
    for attempt in range(retries):
        try:
            response = requests.post(URL, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            return result["data"]["translations"][0]["translatedText"]
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return text  # fallback to original text

def translate_dataframe(df, target_lang):
    """Translate all cells in the dataframe, preserving column names"""
    df_translated = df.copy()
    for col in df.columns:
        with tqdm(total=len(df), desc=f"Translating column '{col}' to {target_lang}") as pbar:
            df_translated[col] = df[col].apply(lambda x: _translate_with_progress(x, target_lang, pbar))
    return df_translated

def _translate_with_progress(text, target_lang, pbar):
    """Helper to update progress bar while translating"""
    translated = translate_text(text, target_lang)
    pbar.update(1)
    return translated

# ----------------- MAIN -----------------
def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    input_path = os.path.join(INPUT_DIRECTORY, INPUT_FILE_NAME)
    df = pd.read_csv(input_path)
    
    for lang in LANGUAGES:
        df_translated = translate_dataframe(df, lang)
        output_file = os.path.join(OUTPUT_DIRECTORY, f"{INPUT_FILE_NAME.split('.csv')[0]}_{lang}.csv")
        df_translated.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Saved translated CSV: {output_file}")

if __name__ == "__main__":
    main()
