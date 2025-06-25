# Thai Sentence Embedding Model Training

This project fine-tunes a multilingual SentenceTransformer model for Thai sentence embeddings using the XNLI dataset and evaluates with the STSB benchmark.

## Features
- Loads and preprocesses Thai XNLI data for similarity learning
- Uses `bert-base-multilingual-cased` as the base model
- Evaluates with STSB (semantic textual similarity benchmark)
- Saves the trained model for downstream Thai NLP tasks

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Quick Start
1. **Clone this repository**
2. **Install dependencies**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Set up Hugging Face token**
   - Get your token from https://huggingface.co/settings/tokens
   - Add it to `.env` as:
     ```
     HF_HUB_TOKEN=your_token_here
     HF_HOME=./hf_cache
     HF_DATASETS_CACHE=./hf_cache/datasets
     HF_TRANSFORMERS_CACHE=./hf_cache/transformers
     ```
4. **Run training**
   ```sh
   python train_thai_embedding.py
   ```

## Output
- The trained model will be saved in `thai_sentence_transformer_final/`
- Intermediate checkpoints in `thai_embedding_model/`

## Usage Example
After training, you can use the model like this:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('thai_sentence_transformer_final')
embeddings = model.encode(["สวัสดีครับ", "ประโยคภาษาไทย"])
```

## Tips & Notes
- Training is CPU/GPU compatible (GPU recommended for speed)
- You can adjust hyperparameters in `train_thai_embedding.py`
- If you stop training, you can resume from the latest checkpoint in `thai_embedding_model/`
- For best results, monitor validation metrics and select the best checkpoint manually if needed
- The script will automatically install missing dependencies and set up the environment if needed

## Troubleshooting
- If you see authentication errors, check your `.env` and Hugging Face token
- If you see argument errors in `SentenceTransformerTrainingArguments`, check your `sentence-transformers` version and remove unsupported arguments
- For Windows users, activate the virtual environment with `.venv\Scripts\activate`

## License
This project is for research and educational use. See LICENSE if provided.
