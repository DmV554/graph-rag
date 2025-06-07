import hydra
from omegaconf import DictConfig, OmegaConf
import os
import docx2txt
from datasets import load_dataset
from openai import OpenAI
from rapidfuzz.fuzz import partial_ratio
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score, hamming_loss, classification_report
import mlflow
import pandas as pd


def load_prompt(path):
    """Read and return the prompt text, expecting a placeholder '{{ }}' for sample insertion."""
    # Hydra changes the working directory, so we use get_original_cwd()
    original_cwd = hydra.utils.get_original_cwd()
    full_path = os.path.join(original_cwd, path)
    return docx2txt.process(full_path)


def openai_query(client, prompt, model_name):
    """Send a prompt to the OpenAI chat API and return the assistant's reply."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(model=model_name, messages=messages)
    return resp.choices[0].message.content


def detect_options(output, possible_options, threshold):
    """Return list of options whose fuzzy match against the output exceeds the threshold."""
    if not output:  # Handle cases where the model returns an empty string
        return []
    return [opt for opt in possible_options
            if partial_ratio(output, opt) >= threshold]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run the classification experiment, now configured with Hydra
    and tracked with MLflow.
    """
    # Configure MLflow
    mlflow.set_tracking_uri("file:./mlruns")  # O usa una URI remota si tienes un servidor MLflow

    # Set or create experiment
    experiment_name = cfg.get("experiment_name", "text_classification_experiment")
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error accessing experiment: {e}")
        # Fallback: create a new experiment with timestamp
        import time
        experiment_name = f"text_classification_{int(time.time())}"
        experiment_id = mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

    # Start an MLflow run
    with mlflow.start_run():
        # Log parameters from the Hydra config
        print("Logging parameters to MLflow...")
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

        # Initialize OpenAI client using config
        client = OpenAI(api_key=cfg.openai.api_key, base_url=cfg.openai.base_url)

        # Load prompt template
        print(f"Loading prompt from: {cfg.prompt_path}")
        context = load_prompt(cfg.prompt_path)

        # Load dataset
        print(f"Loading dataset: {cfg.dataset.name}/{cfg.dataset.config}")
        ds = load_dataset(cfg.dataset.name, cfg.dataset.config, split=cfg.dataset.split)

        texts, true_labels = [], []
        for item in ds:
            texts.append(item['text'])
            lbls = item['labels'] if isinstance(item['labels'], list) else [item['labels']]
            true_labels.append([ds.features['labels'].feature.int2str(l) for l in lbls])

        possible_options = ds.features['labels'].feature.names

        # --- Inference Loop ---
        responses, pred_labels, all_results = [], [], []
        total_docs = len(texts)
        print(f"Starting inference on {total_docs} documents...")

        for i, (text, labels) in enumerate(zip(texts, true_labels)):
            print(f"Processing document {i + 1}/{total_docs}...")

            prompt = context.replace("{{ }}", text)
            output = openai_query(client, prompt, cfg.model_name)
            detected = detect_options(output, possible_options, cfg.similarity_threshold)

            responses.append(output)
            pred_labels.append(detected)
            all_results.append({
                'index': i,
                'true_labels': labels,
                'predicted_labels': detected,
                'model_output': output
            })

        print("\n--- Classification Complete ---\n")

        # --- Evaluation ---
        mlb = MultiLabelBinarizer(classes=possible_options)
        true_bin = mlb.fit_transform(true_labels)
        pred_bin = mlb.transform(pred_labels)

        # Calculate metrics
        correct_predictions = sum(1 for true, pred in zip(true_labels, pred_labels) if set(true) == set(pred))
        accuracy = correct_predictions / total_docs
        micro_f1 = f1_score(true_bin, pred_bin, average='micro')
        macro_f1 = f1_score(true_bin, pred_bin, average='macro')
        micro_jacc = jaccard_score(true_bin, pred_bin, average='micro')
        macro_jacc = jaccard_score(true_bin, pred_bin, average='macro')
        hamming = 1 - hamming_loss(true_bin, pred_bin)

        # Log metrics to MLflow
        print("Logging metrics to MLflow...")
        metrics = {
            "accuracy": accuracy,
            "micro_f1_score": micro_f1,
            "macro_f1_score": macro_f1,
            "micro_jaccard": micro_jacc,
            "macro_jaccard": macro_jacc,
            "hamming_score": hamming
        }
        mlflow.log_metrics(metrics)

        # Print metrics to console
        for name, value in metrics.items():
            print(f"{name.replace('_', ' ').capitalize()}: {value:.4f}")

        # --- Log Artifacts ---
        # Classification report
        report = classification_report(true_bin, pred_bin, target_names=possible_options)
        print("\nClassification report:")
        print(report)
        mlflow.log_text(report, "classification_report.txt")

        # Detailed results per document
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("detailed_results.csv", index=False)
        mlflow.log_artifact("detailed_results.csv")

        print(f"\n✅ Experiment '{experiment_name}' successfully tracked in MLflow.")
        print(f"MLflow UI: mlflow ui --backend-store-uri file:./mlruns")


if __name__ == "__main__":
    main()