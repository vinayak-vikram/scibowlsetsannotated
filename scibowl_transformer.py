from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

print("imports resolved")

import logging
from transformers import logging as hf_logging

hf_logging.set_verbosity_info()      # shows info-level logs (including progress)
hf_logging.enable_default_handler()  # ensures logs appear in stdout
hf_logging.enable_explicit_format()  # better formatting

dataset = load_dataset("json", data_files={"train": "training_data.json"})
print(dataset["train"][0])
print("dataset loaded")

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
data_collator=DataCollatorForSeq2Seq(tokenizer, model=None)

def preprocess(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(outputs, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset["train"].map(preprocess, batched=True)
print("preprocessed")

training_args = Seq2SeqTrainingArguments(
    output_dir="./scibowl_parser",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    save_steps=500,
    save_total_limit=2,
    predict_with_generate=True,
    logging_steps=1,
    logging_dir="./logs",
    disable_tqdm=False,
    report_to="none",
    fp16=True  # m1 gpu supports mixed precision
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

trainer.train()
print("trained successfully")

trainer.save_model("./scibowl_parser")
tokenizer.save_pretrained("./scibowl_parser")
