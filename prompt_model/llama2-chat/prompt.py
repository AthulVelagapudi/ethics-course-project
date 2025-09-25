import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import pandas as pd
import json
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

filename = 'data/{filename}.csv'
questions_file = 'data/questions.json'
chosen_cols_file = 'data/chosen_cols.json'
model_path = "/assets/models/meta-llama-2-chat-13b"

# Load Personas and Questions
df = pd.read_csv(filename)
with open(questions_file, "r") as f:
    questions = json.load(f)
with open(chosen_cols_file, "r") as f:
    chosen_cols = json.load(f)
    
chosen_qsns = {}
for qsn in questions:
    if chosen_cols['chosen_cols'][qsn] == True and questions[qsn]['description'] not in chosen_cols['persona_cols']:
        chosen_qsns[qsn] = questions[qsn]

# Prompt and Persona Setup
region = 'N_REGION_ISO: Region ISO 3166-2'
urban_rural = 'H_URBRURAL: Urban-Rural'
age = 'X003R: Age recoded (6 intervals)'
gender = 'Q260: Sex'
language = 'Q272: Language at home'
marital_status = 'Q273: Marital status'
education_level = 'Q275R: Highest educational level: Respondent (recoded into 3 groups)'
social_class = 'Q287: Social class (subjective)'
general_prompt_en = '''
Imagine you are a {language}-speaking {marital_status} {gender} from {urban_rural} {region}, India. You are in {age} years of age category and have completed {education_level} education level. You consider yourself part of the {social_class}. Answer the following question from this perspective. Others will read what you choose; your goal is to convince them it was chosen from the perspective of the persona described above.
Select exactly one option. Answer ONLY with the number corresponding to the question, followed by the number corresponding to the chosen option. Do NOT repeat the question or any other text.
'''

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# Inference
batch_size = 10
results = []
raw_results = []
respondent_number = 0

for _, row in df.iterrows():
    respondent_number += 1
    general_context = {
        "language": row[language],
        "marital_status": row[marital_status],
        "gender": row[gender],
        "urban_rural": row[urban_rural],
        "region": row[region],
        "age": row[age],
        "education_level": row[education_level],
        "social_class": row[social_class]
    }

    questions = []
    for qsn_key in chosen_qsns:
        qsn_text = chosen_qsns[qsn_key]['questions'][0]
        options_list = chosen_qsns[qsn_key]['options']
        options_text = "".join([f"{idx+1}. {opt} " for idx, opt in enumerate(options_list)])
        questions.append((qsn_key, qsn_text, options_list, options_text))

    respondent_answers = general_context.copy()
    debug_output = {"persona": general_context, "questions": []}

    # === Process in batches ===
    for i in tqdm(range(0, len(questions), batch_size), desc=f"Processing question batches for respondent {respondent_number}"):
        batch = questions[i:i+batch_size]
        user_prompt = ""
        for idx, (_, q_text, _, opts_text) in enumerate(batch, start=1):
            user_prompt += f"Question {idx}: {q_text}\nOptions: {opts_text}\n"
        user_prompt += "\nAnswer ONLY with numbers in format: Q1: <option_number>, Q2: <option_number>, ... Do NOT repeat questions."
        messages = [
            {"role": "system", "content": general_prompt_en.format(**general_context)},
            {"role": "user", "content": user_prompt}
        ]
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            system_content = general_prompt_en.format(**general_context)
            formatted_prompt = f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_prompt} [/INST]"

        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.0,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, 
            do_sample=False 
        )
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        raw_results.append({
            "question_batch": user_prompt,
            "formatted_prompt": formatted_prompt,
            "answer_text": answer_text
        })

        batch_answers = re.findall(r'Q\d+:\s*(\d+)', answer_text)
        for j, (qsn_key, q_text, opts_list, _) in enumerate(batch):
            if j < len(batch_answers):
                ans_idx = int(batch_answers[j]) - 1
                if 0 <= ans_idx < len(opts_list):
                    ans_value = opts_list[ans_idx]
                else:
                    ans_value = "Invalid answer"
            else:
                ans_value = "No answer"
                
            respondent_answers[qsn_key] = ans_value
            debug_output["questions"].append({
                "question_key": qsn_key,
                "question_text": q_text,
                "options": opts_list,
                "answer_id": batch_answers[j] if j < len(batch_answers) else None,
                "answer_value": ans_value
            })

    results.append(respondent_answers)

# Wide-format CSV
results_df = pd.DataFrame(results)
results_df.to_csv("results/survey_answers_wide.csv", index=False)

# Debug JSON
with open("temp/survey_answers_debug.json", "w", encoding="utf-8") as f:
    json.dump(raw_results, f, indent=4, ensure_ascii=False)