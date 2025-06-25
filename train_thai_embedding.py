from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
import os
import sys
import subprocess
import logging
import dotenv

# .env
dotenv.load_dotenv()

# Hugging Face environment variables
os.environ["HF_HUB_TOKEN"] = os.getenv("HF_HUB_TOKEN")
os.environ["HF_HOME"] = os.getenv("HF_HOME")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE")
os.environ["HF_TRANSFORMERS_CACHE"] = os.getenv("HF_TRANSFORMERS_CACHE")





def setup_logging():
    """Set up logger to output info to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def setup_venv_and_install():
    """Ensure required packages are installed."""
    logger = logging.getLogger(__name__)
    try:
        logger.info("Installing required dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "datasets",
                        "sentence-transformers"], check=True)
        logger.info("Dependencies installed successfully")
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        raise


def load_env():
    """Load environment variables from .env file if present."""
    dotenv.load_dotenv()
    # Optionally set Hugging Face environment variables for cache and token
    hf_token = os.getenv("HF_HUB_TOKEN")
    if hf_token:
        os.environ["HF_HUB_TOKEN"] = hf_token
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
    if hf_datasets_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache
    hf_transformers_cache = os.getenv("HF_TRANSFORMERS_CACHE")
    if hf_transformers_cache:
        os.environ["HF_TRANSFORMERS_CACHE"] = hf_transformers_cache


def load_thai_data():
    """Load and preprocess Thai XNLI dataset as 0-1 similarity labels."""
    logger = logging.getLogger(__name__)
    logger.info("Loading Thai XNLI dataset...")
    try:
        xnli = load_dataset("xnli", "th")
        raw_train = xnli["train"]

        # Map: entailment -> 1.0; neutral & contradiction -> 0.0
        mapping = {0: 1.0, 1: 0.0, 2: 0.0}

        train_dataset = Dataset.from_dict(
            {
                "sentence1": raw_train["premise"],
                "sentence2": raw_train["hypothesis"],
                "label": [mapping[label] for label in raw_train["label"]],
            }
        )
        logger.info(f"Loaded {len(train_dataset)} Thai training examples")
        return train_dataset
    except Exception as e:
        logger.error(f"Error loading Thai data: {e}")
        raise


def load_validation_data():
    """Load STS-B validation set normalized to 0-1 range for cosine-similarity evaluator."""
    logger = logging.getLogger(__name__)
    logger.info("Loading STSB validation data...")
    try:
        val_sts = load_dataset("glue", "stsb", split="validation")
        normalized_scores = [score / 5.0 for score in val_sts["label"]]

        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=val_sts["sentence1"],
            sentences2=val_sts["sentence2"],
            scores=normalized_scores,
            main_similarity="cosine",
            name="stsb_eval"
        )
        logger.info(f"Created evaluator with {len(val_sts)} validation examples")
        return evaluator
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        raise


def create_model_and_loss():
    """Create multilingual sentence transformer and cosine-similarity loss."""
    logger = logging.getLogger(__name__)
    model_name = "bert-base-multilingual-cased"
    logger.info(f"Loading model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    train_loss = losses.CosineSimilarityLoss(model=embedding_model)
    return embedding_model, train_loss


def setup_training_args():
    """Configure training arguments for fine-tuning the model."""
    return SentenceTransformerTrainingArguments(
        output_dir="thai_embedding_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        fp16=True,
        eval_steps=500,
        logging_steps=100,
        save_steps=1000,
        load_best_model_at_end=False,  # ปิดการโหลด best model อัตโนมัติ
        metric_for_best_model="eval_stsb_eval_spearman_cosine",
        greater_is_better=True,
        dataloader_drop_last=False,
        learning_rate=2e-5,
    )


def main():
    """Main training process for Thai sentence embeddings."""
    load_env()
    logger = setup_logging()
    setup_venv_and_install()
    logger.info("Starting Thai sentence transformer training...")

    try:
        # Load datasets
        train_dataset = load_thai_data()
        evaluator = load_validation_data()

        # Model and loss
        embedding_model, train_loss = create_model_and_loss()

        # Training arguments
        args = setup_training_args()

        # Trainer
        trainer = SentenceTransformerTrainer(
            model=embedding_model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator
        )

        logger.info("==== Training started ====")
        trainer.train()
        logger.info("==== Training finished ====")

        # Evaluate final model
        logger.info("Running final evaluation...")
        result = evaluator(embedding_model)
        if isinstance(result, dict):
            for metric, value in result.items():
                logger.info(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
        else:
            logger.info(f"Evaluation result: {result}")

        # Save model
        final_model_path = "thai_sentence_transformer_final"
        embedding_model.save(final_model_path)
        logger.info(f"Model saved to: {final_model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
