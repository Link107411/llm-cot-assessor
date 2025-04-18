from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import re
from datetime import datetime
import json

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

CONTEXT_LENGTH = 32000
MAX_TOKENS = 30000
TEMPERATURE = 0.6
TOP_P = 0.95

start_time = datetime.now()
filename = start_time.strftime("%Y-%m-%d_%H-%M-%S")

llm = LLM(
    model=MODEL_ID,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization='bitsandbytes',
    load_format='bitsandbytes',
    max_model_len=CONTEXT_LENGTH,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def extract_boxed_answer(text: str) -> str:
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    return matches[-1].strip() if matches else None

params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
)

total_results = {"correct": 0, "wrong": 0}
choices = ['A', 'B', 'C', 'D']
cots = []
for subject in ['college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics']:
    data = load_from_disk(f'./mmlu_splits/{subject}/test')
    subject_results = {"correct": 0, "wrong": 0}
    for i, d in enumerate(data):
        user_prompt = (
            f"Solve the following multiple choices question:\n{d['question']}\n"
            f"A) {d['choices'][0]}\n"
            f"B) {d['choices'][1]}\n"
            f"C) {d['choices'][2]}\n"
            f"D) {d['choices'][3]}\n"
            "Please reason step by step, and put your final answer within \\boxed{}. For example, if the answer is A, please write \\boxed{A}."
        )

        prompt =f"<｜begin▁of▁sentence｜><｜User｜>{user_prompt}\n<｜Assistant｜><think>"

        output1 = llm.generate(prompt, params)
        
        response1 = output1[0].outputs[0].text + '\nFinal answer: \\boxed{'

        output2 = llm.generate(prompt + response1, params)

        response2 = output2[0].outputs[0].text

        response = response1 + response2

        extracted_answer = extract_boxed_answer(response)

        cots.append({
            "subject": subject,
            "question": user_prompt,
            "CoT": "<think>" + response,
            "extracted_answer": extracted_answer,
            "true_answer": choices[d["answer"]],
        })

        print(prompt + response)
        print(f'\n{subject} Question {i + 1}/{len(data)}')
        print(f'res toks 1: {len(output1[0].outputs[0].token_ids)}')
        print(f'res toks 2: {len(output2[0].outputs[0].token_ids)}')
        print(f'Extracted answer: {extracted_answer}')
        print(f'True answer: {choices[d["answer"]]}')
        if extracted_answer == choices[d["answer"]]:
            total_results["correct"] += 1
            subject_results["correct"] += 1
        else:
            total_results["wrong"] += 1
            subject_results["wrong"] += 1
        print(f"Current subject results: {subject_results}")
        print(f"Current total results: {total_results}")
        print('Spend time:', datetime.now() - start_time)
        print('=' * 20)

with open(f'./mmlu_test/{filename}.json', 'w') as f:
    json.dump(cots, f, indent=4)