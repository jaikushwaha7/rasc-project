import json
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = "models/caption_generator/t5_scene"
DATA = "data/processed/scene_graphs/t5_data.json"
BATCH_SIZE = 16   # Increase if you have strong GPU


def main():
    # -------------------------
    # Setup device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # -------------------------
    # Load dataset
    # -------------------------
    with open(DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)

    # -------------------------
    # Load model + tokenizer
    # -------------------------
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # -------------------------
    # Prepare containers
    # -------------------------
    gts = {}
    res = {}

    # -------------------------
    # Batched generation
    # -------------------------
    logger.info("Generating captions...")
    idx = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
            batch = dataset[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch["input"],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,      # Beam search improves captions
                early_stopping=True
            )

            predictions = tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            for j, pred in enumerate(predictions):
                gts[idx] = [batch["target"][j]]
                res[idx] = [pred]
                idx += 1

    # -------------------------
    # Compute CIDEr
    # -------------------------
    logger.info("Computing CIDEr...")
    cider = Cider()
    cider_score, _ = cider.compute_score(gts, res)

    # -------------------------
    # Compute BLEU
    # -------------------------
    logger.info("Computing BLEU...")
    bleu = Bleu(4)
    bleu_scores, _ = bleu.compute_score(gts, res)
    bleu4 = bleu_scores[3]

    metrics = {
        "CIDEr": float(cider_score),
        "BLEU-4": float(bleu4),
        "num_samples": len(dataset),
        "model": "T5-small"
    }

    # -------------------------
    # Save metrics
    # -------------------------
    with open("models/caption_generator/t5_scene/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Evaluation complete: {metrics}")
    print("\nFinal Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
