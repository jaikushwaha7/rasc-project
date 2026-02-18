import inspect
import json

from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

from src.utils.logger import ExperimentLogger, get_logger
from src.utils.experiment import ExperimentTracker

logger = get_logger(__name__)
MODEL = "t5-small"
DATA = "data/processed/scene_graphs/t5_data.json"
OUT = "models/caption_generator/t5_scene"

def main():
    data = json.load(open(DATA))
    ds = Dataset.from_list(data)

    tokenizer = T5Tokenizer.from_pretrained(MODEL)
    model = T5ForConditionalGeneration.from_pretrained(MODEL)

    def tokenize(batch):
        x = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=128)
        y = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=64)
        x["labels"] = y["input_ids"]
        return x

    ds = ds.map(tokenize, batched=True)
    ds = ds.train_test_split(0.15)

    args = build_training_args()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"]
    )

    trainer.train()
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    metrics = {
    "CIDEr": cider_score,
    "BLEU-4": bleu4,
    "epochs": 5,
    "model": "T5-small"
    }


def build_training_args():
    params = inspect.signature(TrainingArguments.__init__).parameters

    base_args = {
        "output_dir": OUT,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
        "evaluation_strategy": "epoch",
        "save_total_limit": 2,
        "logging_steps": 200,
        "report_to": "none",
    }

    if "evaluation_strategy" not in params:
        if "eval_strategy" in params:
            base_args["eval_strategy"] = base_args.pop("evaluation_strategy")
        elif "evaluate_during_training" in params:
            base_args.pop("evaluation_strategy", None)
            base_args["evaluate_during_training"] = True

    if "per_device_train_batch_size" not in params and "per_gpu_train_batch_size" in params:
        base_args["per_gpu_train_batch_size"] = base_args.pop("per_device_train_batch_size")

    filtered_args = {key: value for key, value in base_args.items() if key in params}
    return TrainingArguments(**filtered_args)



if __name__ == "__main__":
    main()
