import ctranslate2
import glob
import os
import pandas as pd
import peft
import random
import timeit
import urllib
import wandb
import wandb.apis.reports as wr
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline


os.environ["WANDB_PROJECT"] = "Autocompletion_evaluation"
os.environ["WANDB_ENTITY"] = "reviewco"
os.environ["WANDB_USERNAME"] = "keisuke-kamata"

EVALUATION_TABLE_NAME = "Validation Responses"
LATENCY_TABLE_NAME = "Model Latencies"
MODEL_NAME = "Finetuned-Review-Autocompletion"

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

def get_completion_responses_batch(prompts, ft_path, ct2_path):
    # Get completions for each model in batches
    opt_completions = get_opt_completion_batch(prompts)
    ft_completions = get_ft_completion_batch(prompts, ft_path)
    ct2_completions = get_ct2_completion_batch(prompts, ct2_path)

    responses = []
    for opt, ft, ct2 in zip(opt_completions, ft_completions, ct2_completions):
        responses.append({
            "Production": opt,
            "Staging (finetuned)": ft,
            "Staging (compressed)": ct2
        })
    return responses

def get_ct2_completion_batch(prompts, ct2_path):
    generator = ctranslate2.Generator(ct2_path)
    start_tokens_list = [tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt)) for prompt in prompts]
    results = generator.generate_batch(start_tokens_list, max_length=50)

    completions = []
    for i, prompt in enumerate(prompts):
        full_output = tokenizer.decode(results[i].sequences_ids[0])
        output = full_output[len(prompt):] if full_output.startswith(prompt) else full_output
        completions.append(output.strip())
    return completions

def get_huggingface_completion_batch(prompts, model):
    generator = pipeline('text-generation', model=model)
    responses = generator(prompts, max_new_tokens=50)

    completions = []
    for i, prompt in enumerate(prompts):
        full_output = responses[i][0]["generated_text"]
        output = full_output[len(prompt):] if full_output.startswith(prompt) else full_output
        completions.append(output.strip())
    return completions

def get_ft_completion_batch(prompts, ft_path):
    return get_huggingface_completion_batch(prompts, ft_path)

def get_opt_completion_batch(prompts):
    return get_huggingface_completion_batch(prompts, "facebook/opt-125m")

def get_model_comparison_df(prompts, ft_path, ct2_path):
    trimmed_prompts = [
        " ".join(prompt.split()[:random.randint(5,12)])
        for prompt in prompts
    ]

    responses = get_completion_responses_batch(trimmed_prompts, ft_path, ct2_path)
    df = pd.DataFrame(responses)
    df.insert(0, "prompt", trimmed_prompts)

    return df

def get_latency_df(prompts, num_prompts, ft_path, ct2_path):
  prompts = prompts.to_list()[:num_prompts]
  ct2_time = timeit.timeit(lambda: get_ct2_completion_batch(prompts, ct2_path), number=1)
  ft_time = timeit.timeit(lambda: get_ft_completion_batch(prompts, ft_path), number=1)
  opt_time = timeit.timeit(lambda: get_opt_completion_batch(prompts), number=1)

  return pd.DataFrame({
    "Model": ["Production", "Staging (finetuned)", "Staging (compressed)"],
    f"Latency for {num_prompts} instructions": [opt_time, ft_time, ct2_time],
  })



with wandb.init(job_type="model_evaluation") as run:
    staging_model = wandb.use_artifact(f'{os.environ["WANDB_ENTITY"]}/model-registry/{MODEL_NAME}:staging')
    staging_path = staging_model.download()

    staging_model_ct2 = wandb.use_artifact(f'{os.environ["WANDB_ENTITY"]}/model-registry/{MODEL_NAME}:staging-ct2')
    staging_path_ct2 = staging_model_ct2.download()

    instruction_artifact = run.use_artifact(f'{os.environ["WANDB_ENTITY"]}/{os.environ["WANDB_PROJECT"]}/instruction:production')
    instruction_dir = instruction_artifact.download()

    test_files = glob.glob(f"{instruction_dir}/test/*.parquet")
    test_data = pd.concat([pd.read_parquet(path) for path in test_files])
    prompts = test_data.sample(frac=1)["text"][:10]

    wandb.log({
        EVALUATION_TABLE_NAME: get_model_comparison_df(prompts, ft_path=staging_path, ct2_path=staging_path_ct2),
        LATENCY_TABLE_NAME: get_latency_df(prompts, num_prompts=3, ft_path=staging_path, ct2_path=staging_path_ct2)
    })


    # Create Report
    report = wr.Report(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        title='Model Evaluation: Autocompletion Model',
        description="Data and sample predictions to evaluate the staging candidate model for our review autocompletion algorithm."
    )

    report.width = "fluid"

    runsets = [wr.Runset(
        os.environ["WANDB_ENTITY"],
        os.environ['WANDB_PROJECT']
        )]

    report.blocks = [
        wr.TableOfContents(),
        wr.H1("Report Overview"),
        wr.P(
            "This report contains information to evaluate whether potential staging models "
            "should be moved to production. Model Registry admins can use the view of the "
            "Model Registry at the end of this report to move a staging model into production, "
            "using a Webhook automation."
        ),
        wr.Spotify(spotify_id="7KAveXwQ5xzdHT6GDlNIBu"),
        wr.MarkdownBlock("May this staging model earn 5 stars 🙏."),
        wr.HorizontalRule(),
    ]

    pg = wr.PanelGrid(
        runsets=runsets,
        panels=[
        wr.ScalarChart(
            title="Current Min Eval Loss",
            metric="eval/loss",
            groupby_aggfunc="min",
            font_size="large"),

        wr.ScalarChart(
            title="Current Min Train Loss",
            metric="train/loss",
            groupby_aggfunc="min",
            font_size="large"),

        wr.ScalarChart(
            title="Longest Runtime (sec)",
            metric="train/train_runtime",
            groupby_aggfunc="max",
            font_size="large"),

        wr.LinePlot(x='Step',
                    y=['eval/loss'],
                    smoothing_factor=0.8,
                    layout={'w': 24, 'h': 9})
        ]
    )

    report.blocks += [wr.H1("Key Metrics"), pg]

    pg = wr.PanelGrid(
        runsets=runsets,
        panels=[
            wr.WeavePanelSummaryTable(LATENCY_TABLE_NAME, layout={'w': 24, 'h': 12}),
        ])

    report.blocks += [wr.H1("Latency Data for Models"), pg]


    pg = wr.PanelGrid(
        runsets=runsets,
        panels=[
            wr.WeavePanelSummaryTable(EVALUATION_TABLE_NAME, layout={'w': 24, 'h': 12}),
        ])

    report.blocks += [wr.H1("Sample Predictions"), pg]

    report.blocks += [wr.H1("Autocompletion Model in Model Registry"), wr.WeaveBlockArtifact(os.environ["WANDB_ENTITY"], "model-registry", MODEL_NAME)]
    report.save()

    report_creation_msg = f"Report to review: {urllib.parse.quote(report.url, safe=r'/:')}"
    print(report_creation_msg)

    wandb.alert("New Staging Model Evaluated", report_creation_msg)