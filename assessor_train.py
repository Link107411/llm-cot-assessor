from unsloth import FastLanguageModel
import torch

model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

max_seq_length = 15000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from datasets import load_dataset

dataset = load_dataset("json", data_files="mmlu_cot/college_mathematics.jsonl", split="train")

chat_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
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
{}
<|eot_id|>"""

def formatting_func(example):
    return chat_template.format(
        example["question"],
        example["CoT"],
        "Accept" if example["label"] == 1 else "Reject",
    )

dataset = dataset.map(lambda x: {"text": formatting_func(x)})

# print(dataset[0]["text"])

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
    data_collator=DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="<|start_header_id|>assistant<|end_header_id|>\n",
    ),
)

trainer_stats = trainer.train()