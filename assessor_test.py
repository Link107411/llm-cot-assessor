from unsloth import FastLanguageModel
import re

model_id = "./outputs/checkpoint-12"
# model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

CONTEXT_LENGTH = 15000

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = CONTEXT_LENGTH,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

prompt_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Please read the following question and its reasoning process, and determine whether the reasoning is valid.
The question is between <question> and  </question> tags, and the reasoning is between <reasoning> and </reasoning> tags.

<question>
{}
</question>

<reasoning>
{}
</reasoning>

If the reasoning is logically sound and correctly supports the answer to the question, output 'Accept'.
If the reasoning is flawed, inconsistent, or does not properly support the answer, output 'Reject'.
Only output 'Accept' or 'Reject'.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def extract_response(response: str) -> str:
    match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', response, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_question(question: str) -> str:
    # 移除前綴與後綴
    prefix = "Solve the following multiple choices question:\n"
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}. For example, if the answer is A, please write \\boxed{A}."
    # 去掉前後指定內容
    return question[len(prefix): -len(suffix)]


import json
with open("mmlu_test/2025-04-15_14-18-34.json", "r") as f:
    data = json.load(f)

num_accept = 0
num_reject = 0
for i, d in enumerate(data):
    question = extract_question(d["question"])
    cot = d["CoT"]

    prompt = prompt_template.format(question, cot)

    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=64)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(response)
    response = extract_response(response)
    print("\n\nExtracted response:", response)
    print("=" * 50)

    if response == "Accept" or response == "Accept.":
        num_accept += 1
    elif response == "Reject" or response == "Reject.":
        num_reject += 1

    data[i]["assessor_response"] = response

print("Accept:", num_accept)
print("Reject:", num_reject)
print("Total:", num_accept + num_reject, len(data))

with open("mmlu_test/2025-04-15_14-18-34_assessor_response.json", "w") as f:
    json.dump(data, f, indent=4)