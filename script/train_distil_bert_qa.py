import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

# 1 - Config
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./qa-distilbert-small"
BATCH_SIZE = 8
NUM_EPOCHS = 3
MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 500

# # 2 - Dataset
raw_datasets = load_dataset("squad")

if MAX_TRAIN_SAMPLES:
    raw_datasets["train"] = raw_datasets["train"].select(
        range(min(MAX_TRAIN_SAMPLES, len(raw_datasets["train"])))
    )

if MAX_VAL_SAMPLES:
    raw_datasets["validation"] = raw_datasets["validation"].select(
        range(min(MAX_VAL_SAMPLES, len(raw_datasets["validation"])))
    )

# 3 - Model + Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# 4 - Preprocessing (TRAIN)
def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # No answer -> label as CLS
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        # Real answer
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find start/end of context in token space
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # If answer outside this span -> label CLS
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        # Find the exact token positions
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    return tokenized_examples


# 5 - Preprocessing (VALID)
def prepare_validation_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # DON'T pop offset_mapping! Just keep it
    # offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Modify offset_mapping IN PLACE (do not pop!)
        tokenized_examples["offset_mapping"][i] = [
            o if sequence_ids[j] == 1 else None
            for j, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# 6 - Apply preprocessing
train_dataset = raw_datasets["train"].map(
    prepare_train_features,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

valid_dataset_full = raw_datasets["validation"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)

# Create a version for the Trainer (no offset_mapping, no example_id)
columns_for_model = ["input_ids", "attention_mask", "start_positions", "end_positions"]
valid_dataset_for_model = valid_dataset_full.remove_columns(
    [c for c in valid_dataset_full.column_names if c not in columns_for_model]
)

valid_features_for_postproc = valid_dataset_full


# 7 - Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,
    # IMPORTANT: let Trainer remove dataset columns it doesn't need for the model
    remove_unused_columns=False,
    logging_steps=50,
)

# 8 - Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset_for_model,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# 9 - Training
trainer.train()

# 10 - Save model/tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 11 - Inference example
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model=OUTPUT_DIR, tokenizer=OUTPUT_DIR)

example = {
    "context": raw_datasets["validation"][2]["context"],
    "question": raw_datasets["validation"][2]["question"]
}

print("QUESTION:", example["question"])
print("PRED:", qa_pipeline(example))
print("GOLD:", raw_datasets["validation"][2]["answers"])

#-----------------------------------------------------------
#-------------------- Evaluation ---------------------------
##----------------------------------------------------------

import collections
import numpy as np

def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions,
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    all_start_logits, all_end_logits = raw_predictions

    # Map example_id → list of feature indices
    example_id_to_index = {ex["id"]: i for i, ex in enumerate(examples)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(i)

    predictions = {}
    
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        best_score = float("-inf")
        best_text = ""

        for feature_index in features_per_example[example_id]:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            # Get top start/end indices
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_idx in start_indexes:
                for end_idx in end_indexes:
                    # Skip invalid spans
                    if start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
                        continue
                    if end_idx < start_idx or end_idx - start_idx + 1 > max_answer_length:
                        continue
                    if offset_mapping[start_idx] is None or offset_mapping[end_idx] is None:
                        continue

                    start_char = offset_mapping[start_idx][0]
                    end_char = offset_mapping[end_idx][1]

                    # This is the key fix: use the ORIGINAL context + correct offsets
                    score = start_logits[start_idx] + end_logits[end_idx]
                    if score > best_score:
                        best_score = score
                        best_text = context[start_char:end_char]

        predictions[example_id] = best_text.strip() if best_text else ""

    return predictions


#------------------------------------------------------------------------------
#------------------ F1 + EM Calculation Function ------------------------------
#------------------------------------------------------------------------------

def compute_f1(pred, gold):
    """
    pred: string
    gold: list of possible gold answers (strings)
    """
    def normalize(text):
        import re, string
        text = text.lower()
        text = "".join(ch for ch in text if ch not in string.punctuation)
        text = " ".join(text.split())
        return text

    pred = normalize(pred)
    gold = [normalize(g) for g in gold]

    # Exact match
    em = int(pred in gold)

    # F1
    import collections

    def f1_score(pred, gold):
        pred_tokens = pred.split()
        gold_tokens = gold.split()
        common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return (2 * precision * recall) / (precision + recall)

    f1 = max(f1_score(pred, g) for g in gold)
    return em, f1

#------------------------------------------------------------------------------

print("Running model on validation set...")
raw_preds = trainer.predict(valid_dataset_for_model)

print("Postprocessing predictions...")
final_preds = postprocess_qa_predictions(
    examples=raw_datasets["validation"],
    features=valid_features_for_postproc,
    raw_predictions=raw_preds.predictions,
)

total_em = 0
total_f1 = 0

n = len(raw_datasets["validation"])

for ex in raw_datasets["validation"]:
    qid = ex["id"]
    pred = final_preds[qid]
    gold = ex["answers"]["text"]

    em, f1 = compute_f1(pred, gold)
    total_em += em
    total_f1 += f1

print(f"EM: {total_em / n * 100:.2f}%")
print(f"F1: {total_f1 / n * 100:.2f}%")

# Optional: Print sample errors
print("\nSample predictions:")
for i in range(5):
    ex = raw_datasets["validation"][i]  # correct way to get one example dict
    pred = final_preds.get(ex["id"], "")
    print("Q:", ex["question"])
    print("Pred:", pred)
    print("Gold:", ex["answers"]["text"])
    print("-" * 70)