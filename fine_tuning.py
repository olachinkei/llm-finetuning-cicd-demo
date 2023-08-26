import glob
import os
import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
import wandb
from datasets import load_dataset, Dataset
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

os.environ["WANDB_ENTITY"] = "reviewco"
os.environ["WANDB_PROJECT"] = "Autocompletion_evaluation"
os.environ["WANDB_USERNAME"] = "keisuke-kamata"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_WATCH"] = "gradients"
MODEL_NAME = "Finetuned-opt-125m-dolly"
BASE_MODEL = "facebook/opt-125m"


config = {
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": .05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "training": {
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "report_to": "wandb",
        "warmup_steps": 5,
        "max_steps": 50,
        "learning_rate": 2e-4,
        "fp16": True,
        "logging_steps": 5,
        "save_steps": 25,
        "output_dir": 'outputs'
    },
    "dataset_version": "downsampled"
}

run = wandb.init(config=config, job_type="training")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    device_map='auto',
)

# Prepare data
reviews_artifact = run.use_artifact(f'{os.environ["WANDB_ENTITY"]}/{os.environ["WANDB_PROJECT"]}/reviews:{wandb.config.dataset_version}')
reviews_dir = reviews_artifact.download()

train_files = glob.glob(f"{reviews_dir}/train/*.parquet")
train_data_df = pd.concat([pd.read_parquet(path) for path in train_files])
valid_files = glob.glob(f"{reviews_dir}/valid/*.parquet")
valid_data_df = pd.concat([pd.read_parquet(path) for path in valid_files])

def pad_or_truncate_sequences(sequences, max_length):
    return [list(seq)[:max_length] + [tokenizer.pad_token_id] * (max_length - len(seq)) for seq in sequences]

max_length = 512
padded_and_truncated_input_ids_train = pad_or_truncate_sequences(train_data_df['input_ids'].tolist(), max_length)
padded_and_truncated_input_ids_valid = pad_or_truncate_sequences(valid_data_df['input_ids'].tolist(), max_length)

train_encodings = {
    'input_ids': torch.tensor(padded_and_truncated_input_ids_train, dtype=torch.long),
    'attention_mask': torch.tensor(pad_or_truncate_sequences(train_data_df['attention_mask'].tolist(), max_length), dtype=torch.long),
    'labels': torch.tensor(padded_and_truncated_input_ids_train, dtype=torch.long)
}

valid_encodings = {
    'input_ids': torch.tensor(padded_and_truncated_input_ids_valid, dtype=torch.long),
    'attention_mask': torch.tensor(pad_or_truncate_sequences(valid_data_df['attention_mask'].tolist(), max_length), dtype=torch.long),
    'labels': torch.tensor(padded_and_truncated_input_ids_valid, dtype=torch.long)
}

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_data = CustomDataset(train_encodings)
valid_data = CustomDataset(valid_encodings)


# Setup for LoRa
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

lora_config = LoraConfig(**wandb.config["lora_config"])
model = get_peft_model(model, lora_config)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=valid_data,
    args=transformers.TrainingArguments(**wandb.config["training"]),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# Training
trainer.train()
wandb.finish() # ensure model uploaded before model registry checks



def convert_qlora2ct2(adapter_path,
                      full_model_path="opt125m-finetuned",
                      offload_path="opt125m-offload",
                      ct2_path="opt125m-finetuned-ct2",
                      quantization="int8"):

    peft_model_id = adapter_path
    peftconfig = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForCausalLM.from_pretrained(
      BASE_MODEL,
      offload_folder  = offload_path,
      device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(model, peft_model_id)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(full_model_path)
    tokenizer.save_pretrained(full_model_path)

    if quantization == False:
      os.system(f"ct2-transformers-converter --model {full_model_path} --output_dir {ct2_path} --force")
    else:
      os.system(f"ct2-transformers-converter --model {full_model_path} --output_dir {ct2_path} --quantization {quantization} --force")
    return ct2_path


"""
Link to model registry
with wandb.init() as run:
    api = wandb.Api()
    runs = api.runs()

    summaries = [pd.Series(dict(run.summary), name=run.id) for run in runs]
    run_with_best_model = pd.concat(summaries, axis=1).transpose()['eval/loss'].sort_values().index[0]
    best_model_adapters = wandb.use_artifact(f'{os.environ["WANDB_ENTITY"]}/{os.environ["WANDB_PROJECT"]}/model-{run_with_best_model}:latest')

    adapter_path = best_model_adapters.download()
    ct2_model_dir = convert_qlora2ct2(adapter_path)

    best_model_ft = wandb.Artifact(f"model-{run_with_best_model}-ft", type="model")
    best_model_ft.add_dir("opt125m-finetuned")
    run.log_artifact(best_model_ft)
    run.link_artifact(best_model_ft, f'{os.environ["WANDB_ENTITY"]}/model-registry/{MODEL_NAME}', aliases=['staging'])

    best_model_ct2 = wandb.Artifact(f"model-{run_with_best_model}-ct2", type="model")
    best_model_ct2.add_dir(ct2_model_dir)
    run.log_artifact(best_model_ct2)
    run.link_artifact(best_model_ct2, f'{os.environ["WANDB_ENTITY"]}/model-registry/{MODEL_NAME}', aliases=['staging-ct2'])
"""