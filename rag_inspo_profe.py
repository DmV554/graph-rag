import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import pandas as pd
from datasets import load_dataset
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, jaccard_score, hamming_loss, classification_report
from rapidfuzz.fuzz import partial_ratio
from tqdm import tqdm

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # --- Logging & MLflow Setup ---
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():
        # Log configuration
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # --- Load prompt template from TXT ---
        original_cwd = hydra.utils.get_original_cwd()
        prompt_path = os.path.join(original_cwd, cfg.rag.prompt_path)
        if not os.path.exists(prompt_path):
            logger.error(f"Prompt file not found: {prompt_path}")
            sys.exit(1)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template_text = f.read()

        # --- Load dataset via `datasets` library ---
        ds = load_dataset(cfg.dataset.name, cfg.dataset.config, split=cfg.dataset.split)
        if cfg.dataset.eval_samples:
            ds = ds.select(range(cfg.dataset.eval_samples))

        # Extract label names from dataset features
        label_field = cfg.dataset.label_field
        text_field = cfg.dataset.text_field
        label_names = ds.features[label_field].feature.names

        # --- Initialize LLM & Embeddings via LlamaIndex ---
        Settings.llm = Ollama(model=cfg.rag.llm_model, request_timeout=cfg.rag.request_timeout)
        Settings.embed_model = HuggingFaceEmbedding(model_name=cfg.rag.embed_model)

        # --- Load persisted index ---
        persist_dir = cfg.rag.persist_dir
        if not os.path.exists(persist_dir):
            logger.error(f"Index directory not found: {persist_dir}")
            sys.exit(1)
        storage = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage)

        # --- Build Query Engine ---
        qe = index.as_query_engine(
            similarity_top_k=cfg.rag.similarity_top_k,
            response_mode=cfg.rag.response_mode
        )

        # --- Prepare classification prompt template ---
        categories_str = "\n".join([f"- {l}" for l in label_names])
        full_template = prompt_template_text.replace("{categorias}", categories_str)
        tpl = PromptTemplate(full_template)
        qe.update_prompts({"response_synthesizer:text_qa_template": tpl})

        # --- Inference ---
        texts = ds[text_field]
        true_indices = ds[label_field]
        mlabels = MultiLabelBinarizer(classes=label_names)
        y_true, y_pred = [], []
        detailed = []

        for text, true_idx in tqdm(zip(texts, true_indices), total=len(texts)):
            y_true.append(true_idx)
            res = qe.query(text)
            out = res.response.strip()
            preds = [lab for lab in label_names if partial_ratio(out, lab) >= cfg.rag.match_threshold]
            y_pred.append(preds)
            detailed.append({
                'text': text,
                'true_idx': true_idx,
                'pred_names': preds,
                'model_output': out
            })

        # --- Metrics ---
        y_true_names = [[label_names[i] for i in idx] for idx in y_true]
        y_true_bin = mlabels.fit_transform(y_true_names)
        y_pred_bin = mlabels.transform(y_pred)

        accuracy = sum(set(t)==set(p) for t,p in zip(y_true_names, y_pred)) / len(y_true)
        micro_f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
        macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
        micro_j = jaccard_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
        macro_j = jaccard_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
        hamming = 1 - hamming_loss(y_true_bin, y_pred_bin)

        metrics = {
            'accuracy': accuracy,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'micro_jaccard': micro_j,
            'macro_jaccard': macro_j,
            'hamming_score': hamming
        }
        mlflow.log_metrics(metrics)

        # --- Reports & Artifacts ---
        report = classification_report(y_true_bin, y_pred_bin, target_names=label_names, zero_division=0)
        logger.info("\nClassification Report:\n%s", report)
        mlflow.log_text(report, "classification_report.txt")

        df_out = pd.DataFrame(detailed)
        df_out.to_csv(cfg.dataset.output_csv, index=False)
        mlflow.log_artifact(cfg.dataset.output_csv)

        logger.info("Run complete. Metrics: %s", metrics)

if __name__ == '__main__':
    main()
