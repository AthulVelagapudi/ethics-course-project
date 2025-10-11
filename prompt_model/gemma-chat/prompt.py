import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import re
import json
import warnings
import logging
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

language_code = 'te'

# ==============================
# CONFIGURATION
# ==============================
FILENAME = f'india/2022/2022_india_persona_groups_cleaned_te copy'
MODEL_PATH = "/assets/models/google-gemma-3-it-12b"


states_in_language = {
    'en': {
        "bengal": "West Bengal",
        'telangana': "Telangana",
        'maharashtra': "Maharashtra",
        'punjab': "Punjab",
        'delhi': "Delhi",
        'bihar': "Bihar",
        'haryana': "Haryana",
        'up': "Uttar Pradesh"
    },
    'bn': {
        "bengal": "পশ্চিমবঙ্গ"
    },
    'te': {
        'telangana': "తెలంగాణ"
    }
}

# ==============================
# PERSONA COLUMNS
# ==============================
region = 'N_REGION_ISO: Region ISO 3166-2'
urban_rural = 'H_URBRURAL: Urban-Rural'
age = 'X003R: Age recoded (6 intervals)'
gender = 'Q260: Sex'
language = 'Q272: Language at home'
marital_status = 'Q273: Marital status'
education_level = 'Q275R: Highest educational level: Respondent (recoded into 3 groups)'
social_class = 'Q287: Social class (subjective)'

# ==============================
# LOAD DATA
# ==============================
df_chosen = pd.read_csv(f"data/{FILENAME}.csv")
print(f"Rows: {df_chosen.shape[0]}, Columns: {df_chosen.shape[1]}")

with open(f"data/questions.json", "r") as f:
    questions = json.load(f)
with open("data/chosen_cols_updated.json", "r") as f:
    chosen_cols = json.load(f)

chosen_qsns = {
    qsn: questions[qsn]
    for qsn in questions
    if chosen_cols['chosen_cols'][qsn]
    and questions[qsn]['description'] not in chosen_cols['persona_cols']
}

#df_chosen = df_whole[df_whole[region].str.contains(states_in_language[language_code][state], case=False, na=False)].copy()
print("Length:", len(df_chosen))


# ==============================
# LANGUAGE PROMPTS
# ==============================
general_prompts = {
    'en': '''
    Imagine you are a {language}-speaking {marital_status} {gender} from {urban_rural} {region}, India. You are in {age} years of age category and have completed {education_level} education level. You consider yourself part of the {social_class}. Answer the following question from this perspective. Others will read what you choose; your goal is to convince them it was chosen from the perspective of the persona described above.
    Select exactly one option. Answer ONLY with the number corresponding to the question, followed by the number corresponding to the chosen option. Do NOT repeat the question or any other text.
    ''',
    'bn': '''
    কল্পনা করুন আপনি একজন {language}-ভাষী {marital_status} {gender}, যে {urban_rural} {region}, ভারতের অধিবাসী। আপনার বয়স {age} বছর এবং আপনি {education_level} শিক্ষাগত স্তর সম্পন্ন করেছেন। আপনি নিজেকে {social_class} সামাজিক শ্রেণীর অংশ মনে করেন। নিম্নলিখিত প্রশ্নের উত্তর এই দৃষ্টিকোণ থেকে দিন। অন্যরা আপনার নির্বাচিত উত্তর দেখবে; আপনার লক্ষ্য হলো তাদের বিশ্বাস করানো যে এটি উপরোক্ত বর্ণিত ব্যক্তিত্বের দৃষ্টিকোণ থেকে নির্বাচন করা হয়েছে।
    নির্দিষ্টভাবে ঠিক একটি বিকল্প নির্বাচন করুন। শুধুমাত্র প্রশ্নের সংখ্যা এবং নির্বাচিত বিকল্পের সংখ্যার সঙ্গে উত্তর দিন। প্রশ্ন বা অন্য কোনো লেখা পুনরায় লিখবেন না।
    ''',
    'te': '''
    కాల్పనికంగా మీరు {language}-భాష మాట్లాడే {marital_status} {gender}, {urban_rural} {region}, భారతదేశం నుండి ఉన్నారని ఊహించుకోండి. మీరు {age} వయసు విభాగంలో ఉన్నారు మరియు {education_level} విద్యా స్థాయిని పూర్తి చేసారు. మీరు మీరును {social_class} సామాజిక తరగతి భాగంగా భావిస్తున్నారు. ఈ దృక్కోణం నుండి కింది ప్రశ్నకు జవాబు చెప్పండి. ఇతరులు మీరు ఎంచుకున్నదాన్ని చదువుతారు; మీ లక్ష్యం వారు నమ్మడానికి ఇది పైగా వర్ణించబడిన వ్యక్తిత్వం దృక్కోణం నుండి ఎంచుకోబడినదని చూపించడం.
    నిర్దిష్టంగా ఒక్క ఆప్షన్ మాత్రమే ఎంచుకోండి. కేవలం ప్రశ్న సంఖ్య మరియు ఎంచుకున్న ఆప్షన్ సంఖ్యతో జవాబు ఇవ్వండి. ప్రశ్న లేదా ఇతర ఏదైనా వచనం పునరావృతం చేయవద్దు.
    ''',
    'hi': '''
    कल्पना कीजिए कि आप भारत के {urban_rural} क्षेत्र {region} से एक {language}-भाषी {marital_status} {gender} हैं। आप {age} वर्ष की आयु श्रेणी में आते हैं और आपने {education_level} स्तर की शिक्षा पूरी की है। आप स्वयं को {social_class} वर्ग का हिस्सा मानते हैं। नीचे दिए गए प्रश्न का उत्तर इसी दृष्टिकोण से दें। अन्य लोग आपका उत्तर पढ़ेंगे; आपका उद्देश्य उन्हें यह विश्वास दिलाना है कि यह उत्तर ऊपर वर्णित व्यक्ति के दृष्टिकोण से चुना गया है। सिर्फ एक विकल्प चुनें। केवल प्रश्न के क्रमांक के बाद चुने गए विकल्प के क्रमांक के साथ उत्तर दें। प्रश्न या कोई अन्य पाठ दोहराएँ नहीं।
    '''
}

user_prompts = {
    'en': "\nAnswer ONLY with numbers in this format: Q1: <option_number>, Q2: <option_number>, ... Do NOT repeat the questions or any other text.",
    'bn': "\nশুধুমাত্র সংখ্যার মাধ্যমে উত্তর দিন, এই ফরম্যাটে: Q1: <option_number>, Q2: <option_number>, ... প্রশ্ন বা অন্য কোনো লেখা পুনরায় লিখবেন না।",
    'te': "\nకేవలం సంఖ్యలతో మాత్రమే జవాబు ఇవ్వండి, ఈ ఫార్మాట్‌లో: Q1: <option_number>, Q2: <option_number>, ... ప్రశ్నలు లేదా ఇతర వచనం పునరావృతం చేయవద్దు.",
    'hi': "\nकेवल संख्याओं में उत्तर दें, इस प्रारूप में: Q1: <विकल्प_संख्या>, Q2: <विकल्प_संख्या>, ... प्रश्नों या किसी अन्य पाठ को दोहराएँ नहीं।"
}

joining_prompts = {
    'en': "Question {idx}: {q_text}\nOptions: {opts_text}\n",
    'bn': "প্রশ্ন {idx}: {q_text}\nবিকল্পসমূহ: {opts_text}\n",
    'te': "ప్రశ్న {idx}: {q_text}\nఆప్షన్లు: {opts_text}\n",
    'hi': "प्रश्न {idx}: {q_text}\nविकल्प: {opts_text}\n"
}

# ==============================
# MODEL INITIALIZATION
# ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
print("Model loaded successfully.")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==============================
# RESPONSE GENERATION
# ==============================
def find_responses(df, state, tokenizer, model, chosen_cols, language_code='en'):
    with open(f"data/translated_questions/questions_{language_code}.json", "r") as f:
        questions = json.load(f)
    with open("data/chosen_cols_updated.json", "r") as f:
        chosen_cols = json.load(f)

    chosen_qsns = {}

    for qsn in questions:
        if chosen_cols['chosen_cols'][qsn] == True and questions[qsn]['description'] not in chosen_cols['persona_cols']:
            chosen_qsns[qsn] = questions[qsn]

    batch_size = 20
    results = []
    raw_results = []
    respondent_number = 0
    #print(len(chosen_qsns))
    answer_pattern = re.compile(r'Q\s*(\d+)\s*[:\-]\s*([0-9]+)', re.IGNORECASE)

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
            options_list = chosen_qsns[qsn_key]['options']
            options_text = "".join([f"{idx+1}. {opt} " for idx, opt in enumerate(options_list)])
        
            # Add all 4 question variants (0, 1, 2, 3)
            for qsn_variant in range(4):
                if qsn_variant < len(chosen_qsns[qsn_key]['questions']):
                    qsn_text = chosen_qsns[qsn_key]['questions'][qsn_variant]
                    questions.append((qsn_key, qsn_text, options_list, options_text, qsn_variant))
                else:
                # If there are fewer than 4 variants, break
                    break


        respondent_answers = general_context.copy()
        debug_output = {"persona": general_context, "questions": []}
        
        for i in tqdm(range(0, len(questions), batch_size),
                      desc=f"Processing question batches for respondent {respondent_number}"):
            batch = questions[i:i + batch_size]
            
            general_prompt = general_prompts[language_code].format(**general_context)
            user_prompt = ""
            for idx, (_, q_text, _, opts_text, _) in enumerate(batch, start=1):
                user_prompt += joining_prompts[language_code].format(idx=idx, q_text=q_text, opts_text=opts_text)
            user_prompt += user_prompts[language_code]

            messages = [
                {"role": "system", "content": general_prompt},
                {"role": "user", "content": user_prompt}
            ]
            if hasattr(tokenizer, 'apply_chat_template'):
                formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            else:
                system_content = general_prompt.format(**general_context)
                formatted_prompt = f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_prompt} [/INST]"

            tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
            if tokenizer_max_len is None or tokenizer_max_len > 100000:
                max_input_len = 2048  # Safe default
            else:
                max_input_len = min(tokenizer_max_len, 2048)  # Cap at 2048 for safety

            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=max_input_len, padding=True).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
            prompt_len = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0, prompt_len:]
            answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            raw_results.append({"question_batch": user_prompt, "answer_text": answer_text})
            # print("--- RAW MODEL OUTPUT ---")
            # print(answer_text)
            # print("------------------------")
            # break
            matches = answer_pattern.findall(answer_text)
            # matches -> list of (qnum_str, ansnum_str)
            answers_by_qnum = {int(q): int(a) for q, a in matches}

        # now map answers to the batch questions
            for j, (qsn_key, q_text, opts_list, _, qsn_variant) in enumerate(batch):
                q_num_in_batch = j + 1 
                if q_num_in_batch in answers_by_qnum:
                    ans_idx = answers_by_qnum[q_num_in_batch] - 1
                    if 0 <= ans_idx < len(opts_list):
                        ans_value = opts_list[ans_idx]
                        ans_id = answers_by_qnum[q_num_in_batch]
                    else:
                        ans_value = "Invalid answer"
                        ans_id = answers_by_qnum[q_num_in_batch]
                else:
                    ans_value = "No answer"
                    ans_id = None

                variant_key = f"{qsn_key}_variant_{qsn_variant}"
                respondent_answers[variant_key] = ans_value
                debug_output["questions"].append({
                "question_key": qsn_key,
                "question_variant": qsn_variant,
                "question_text": q_text,
                "options": opts_list,
                "answer_id": ans_id,
                "answer_value": ans_value
            })

        results.append(respondent_answers)
        if respondent_number % 10 == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv("/home/athul/Persona/ethics-course-project/responses/gemma/survey_answers_wide_gemma_telugu.csv", index=False)
        results_df = pd.DataFrame(results)
        results_df.to_csv("/home/athul/Persona/ethics-course-project/responses/gemma/gemma_survey_answers_wide_telugu.csv", index=False)


# ==============================
# AGGREGATE MAJORITY ANSWERS
# ==============================
def get_most_frequent_answers(state, language_code='en'):
    df = pd.read_csv(f"/home/athul/Persona/ethics-course-project/responses/gemma/gemma_survey_answers_wide_telugu.csv")
    question_cols = [col for col in df.columns if ' - ' in col and col.split(' - ')[0].startswith('Q')]
    question_prefixes = sorted(set(col.split(' - ')[0] for col in question_cols))

    for q in question_prefixes:
        q_cols = [col for col in df.columns if col.startswith(q + ' -')]
        df[q] = df[q_cols].apply(lambda row: row.mode().iloc[0] if not row.mode().empty else None, axis=1)
        df.drop(columns=q_cols, inplace=True)

    df.to_csv(f"Persona/ethics-course-project/responses/gemma/most_frequent_answers_{state}_{language_code}.csv", index=False)


# ==============================
# MAIN
# ==============================
def main():
    find_responses(df_chosen, "telangana", tokenizer, model, chosen_cols, language_code)
    get_most_frequent_answers("telangana", language_code)

if __name__ == "__main__":
    main()
