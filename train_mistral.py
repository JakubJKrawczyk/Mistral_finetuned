# mistral_raw_finetune.py
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Konfiguracja
MODEL_NAME = "unsloth/Mistral-Small-Instruct-2409"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./model"
DATA_DIR = "./data"

# Załaduj model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_8bit=True,
    
)

# LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        report_to="none",
    ),
)

# Wczytaj surowy tekst
for file in dir(DATA_DIR):
    with open(file, "r") as f:

        lines = [line.strip() for line in f if line.strip()]

        # Utwórz dataset
        dataset = Dataset.from_dict({"text": lines})

        trainer.train_dataset = dataset

        print(f"Załadowano {len(lines)} obserwacji")

        # Trening

        print("Trening...")
        trainer.train()

# Zapisz
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"Model: {OUTPUT_DIR}/final")

# Test
FastLanguageModel.for_inference(model)
inputs = tokenizer(["Podaj poradę dotyczącą"], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print("\nTest:")
print(tokenizer.decode(outputs[0]))
