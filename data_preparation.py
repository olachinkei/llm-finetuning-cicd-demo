import glob
import os
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

os.environ["WANDB_ENTITY"] = "reviewco"
os.environ["WANDB_PROJECT"] = "Autocompletion with evaluation"
os.environ["WANDB_USERNAME"] = "keisuke-kamata"

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
data = load_dataset("databricks-dolly-15k")
data = data.map(lambda samples: tokenizer(samples['text']), batched=True)

INITIAL_UPLOAD_DATE = "20230826"
train_shuffled = data["train"].shuffle(seed=42)

total_train_samples = len(train_shuffled)
train_data_size = int(0.9 * total_train_samples)
valid_data_size = total_train_samples - train_data_size

train_data = train_shuffled.select(range(0, train_data_size))
valid_data = train_shuffled.select(range(train_data_size, total_train_samples))
test_data = data["test"].shuffle(seed=42)

train_data.to_pandas().to_parquet(f"instruction_data/train/{INITIAL_UPLOAD_DATE}.parquet", index=False)
valid_data.to_pandas().to_parquet(f"instruction_data/valid/{INITIAL_UPLOAD_DATE}.parquet", index=False)
test_data.to_pandas().to_parquet(f"instruction_data/test/{INITIAL_UPLOAD_DATE}.parquet", index=False)

with wandb.init() as run:
    artifact = wandb.Artifact('instruction', type='dataset')
    artifact.add_dir("instruction_data/")
    run.log_artifact(artifact, aliases=["production"])

for file in glob.glob("instruction_data/*.parquet"):
    os.remove(file)

train_data = train_shuffled.select(range(0, 200))
valid_data = train_shuffled.select(range(200, 300))

train_data.to_pandas().to_parquet(f"instruction_data/train/{INITIAL_UPLOAD_DATE}.parquet", index=False)
valid_data.to_pandas().to_parquet(f"instruction_data/valid/{INITIAL_UPLOAD_DATE}.parquet", index=False)

with wandb.init() as run:
    artifact = wandb.Artifact('instruction', type='dataset')
    artifact.add_dir("instruction_data/")
    run.log_artifact(artifact, aliases=["downsampled"])
