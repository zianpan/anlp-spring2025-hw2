import pandas as pd
import json
import threading
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
import torch
import re
from tqdm import tqdm, trange
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

gc.collect()
torch.cuda.empty_cache()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
df = pd.read_csv('../data/texts_urls_filtered.csv')

model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "unsloth/phi-4-bnb-4bit"
custom_cache_dir = "/mnt/new_volume"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
tokenizer.pad_token = tokenizer.eos_token  # set the pad token to be the same as the eos token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.float16,
    device_map = device,
    cache_dir=custom_cache_dir
)


def generate_qa_pairs(document_text, model, tokenizer, num_pairs=10, use_streamer=True):
    prompt = f"""
    Generate {num_pairs} high-quality question-answer pairs based on the given document about Pittsburgh and Carnegie Mellon University (CMU).

    # from the following document:
    
    # ---
    # {document_text}
    # ---
    
    **Requirements:**
    - Ensure questions are diverse in type, including but not limited to:
    - Historical facts
    - Notable events
    - Landmarks and locations
    - University-specific information
    - Cultural or entertainment-related facts
    - Important figures related to CMU or Pittsburgh
    - Each question should have a **direct answer** based on the content provided.
    - Ensure questions are **concise** and **clear**.
    - Answers should be **factual** and **direct**.
    - Avoid simple question like "What is the name of"

    **Output Format:**
    Return a JSON-formatted list where each entry contains:
    - `"question"`: The generated question.
    - `"answer"`: The corresponding answer.

    **Example Output:**

    [
        {{"question": "Who is Pittsburgh named after?", "answer": "William Pitt"}},
        {{"question": "What year was Carnegie Mellon University founded?", "answer": "1900"}},
        {{"question": "Which bridge in Pittsburgh is famously yellow?", "answer": "Roberto Clemente Bridge"}},
        {{"question": "Which famous AI professor at CMU co-founded Duolingo?", "answer": "Luis von Ahn"}},
        {{"question": "Who hosts the Burgh Bus comedy tour in Pittsburgh?", "answer": "Matt Light."}}
    ]

    Now it's your turn.

    Output:
    
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    if use_streamer:
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=350,
            do_sample=True,
            streamer=streamer
        )
        
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        generated_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text
        thread.join()
        
    else:
        

        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # print(generated_text)
    
    pattern = r'\[.*?\]'
    matches = re.findall(pattern, generated_text, flags=re.DOTALL)

    error = None
    if len(matches) <= 1:
        # print("No QA pairs generated")
        error = (generated_text, document_text)
    
    del generated_text
    del document_text
    del inputs
    del prompt
    
    return matches, error
    
    # try:
    #     qa_pairs = json.loads(matches[0]) if matches else []
    # except json.JSONDecodeError:
    #     qa_pairs = []
    
    # error = None if qa_pairs else (generated_text, document_text)
    
    # return qa_pairs, error

# %%
import random
# add random seed
random.seed(42)

idx = list(range(0, len(df)))
# shuffle the list
random.shuffle(idx)


# %%
all_matches = []
error_responses = []

# %%
for i in trange(100,700):
    document_text = df.iloc[idx[i]]['TEXT']
    url = df.iloc[idx[i]]['URL']
    if len(document_text) > 40000:
        continue
    qa_pairs_matches, error = generate_qa_pairs(document_text, model, tokenizer, num_pairs=5, use_streamer=False)
    # print(qa_pairs_matches)
    if error:
        error_responses.append(error)

    all_matches.append((url, qa_pairs_matches))

# %%
# create a pd dataframe from a list of json objects

questions = []
answers = []
url_li = []

wrong_json = []

for url, sub_list in all_matches:
    for sub_sub_list in sub_list:
        try:
            sub_sub_list = json.loads(sub_sub_list)
        except:
            wrong_json.append(sub_sub_list)
            continue
        for qa in sub_sub_list:
            if type(qa) != dict:
                continue
            if 'question' not in qa or 'answer' not in qa:
                continue

            questions.append(qa['question'])
            answers.append(qa['answer'])
            url_li.append(url)

qa_df = pd.DataFrame({'url': url_li, 'question': questions, 'reference_answer': answers})

qa_df.drop_duplicates(subset=['question'], inplace=True)

qa_df.to_csv('qa_pairs_new_600_38.csv', index=False)
