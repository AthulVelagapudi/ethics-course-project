import os
import json
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
INPUT_DIRECTORY = "data"
INPUT_FILE_NAME = "2022_indian_majority_answers_by_persona"
OUTPUT_DIRECTORY = "data/translated_data"
LANG_CODE = "te"

# === SETUP ===
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

with open("data/translated_questions/questions_en.json", "r", encoding="utf-8") as f:
    questions_en = json.load(f)
with open(f"data/translated_questions/questions_{LANG_CODE}.json", "r", encoding="utf-8") as f:
    questions_bn = json.load(f)

input_path = os.path.join(INPUT_DIRECTORY, f"{INPUT_FILE_NAME}.csv")
df = pd.read_csv(input_path)

# Identify columns that correspond to questions
question_cols = [col for col in df.columns if col.split(":")[0] in questions_bn]
print(f"Columns to translate ({len(question_cols)}): {question_cols}")

# Translation logic
def translate_answer(qcode, answer):
    """Map English option to Bengali equivalent or keep unchanged if scale question."""
    if not isinstance(answer, str) or not answer.strip():
        return answer

    q_en = questions_en.get(qcode)
    q_bn = questions_bn.get(qcode)

    if not q_en or not q_bn:
        return answer

    # If scale-based question, keep answer unchanged
    if q_en.get("scale", False):
        return answer

    # Find index of answer in English options
    try:
        idx = q_en["options"].index(answer.strip())
        return q_bn["options"][idx]
    except (ValueError, IndexError):
        return answer


# Apply translation
for col in tqdm(question_cols, desc="Translating columns"):
    qcode = col.split(":")[0]
    df[col] = df[col].apply(lambda x: translate_answer(qcode, x))

# Save translated output
output_path = os.path.join(OUTPUT_DIRECTORY, f"{INPUT_FILE_NAME}_{LANG_CODE}.csv")
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ Saved translated CSV: {output_path}")
