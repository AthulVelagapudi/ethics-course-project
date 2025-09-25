import os
from dotenv import load_dotenv
import requests
import json
import time
from tqdm import tqdm

INPUT_DIRECTORY = "data"
INPUT_FILE_NAME = "questions"
OUTPUT_DIRECTORY = "data/translated_questions"
LANGUAGES = ["mr", "pa"]

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
url = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"

def count_strings(obj):
    """Count total number of strings in JSON-like object"""
    if isinstance(obj, dict):
        return sum(count_strings(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_strings(elem) for elem in obj)
    elif isinstance(obj, str):
        return 1
    else:
        return 0
    
def translate_text(text, target_lang, retries=3):
    """Translate a single text string using Google Translate API with retries"""
    data = {
        "q": text,
        "target": target_lang,
        "format": "text"
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            return result["data"]["translations"][0]["translatedText"]
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return text

def translate_json(obj, target_lang, pbar=None):
    """Recursively translate all string fields in JSON-like object"""
    if isinstance(obj, dict):
        return {k: translate_json(v, target_lang, pbar) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [translate_json(elem, target_lang, pbar) for elem in obj]
    elif isinstance(obj, str):
        if pbar:
            pbar.update(1)
        return translate_text(obj, target_lang)
    else:
        return obj
    
def main():
    with open(f"{INPUT_DIRECTORY}/{INPUT_FILE_NAME}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    for lang in LANGUAGES:
        total = count_strings(data)
        with tqdm(total=total, desc="Translating") as pbar:
            translated_data = translate_json(data, lang, pbar)
        with open(os.path.join(OUTPUT_DIRECTORY, f"{INPUT_FILE_NAME}_{lang}.json"), "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
    