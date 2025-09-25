import os
from dotenv import load_dotenv
import requests
import json

INPUT_FILE_NAME = "../data/questions"
OUTPUT_DIRECTORY = "../data/translated_questions"
LANGUAGES = ["hi", "bn", "mr", "te", "pa"]

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
url = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"

def translate_text(text, target_lang):
    """Translate text to target language using Google Translate API"""
    data = {
        "q": text,
        "target": target_lang,
        "format": "text"
    }
    response = requests.post(url, data=data)
    result = response.json()
    return result["data"]["translations"][0]["translatedText"]

def translate_json(obj, target_lang):
    """Recursively translate all string fields in JSON-like object"""
    if isinstance(obj, dict):
        return {k: translate_json(v, target_lang) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [translate_json(elem, target_lang) for elem in obj]
    elif isinstance(obj, str):
        return translate_text(obj, target_lang)
    else:
        return obj 
    
def main():
    with open(f"{INPUT_FILE_NAME}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    for lang in LANGUAGES:
        translated_data = translate_json(data, lang)
        with open(os.path.join(OUTPUT_DIRECTORY, f"{INPUT_FILE_NAME}_{lang}.json"), "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
    