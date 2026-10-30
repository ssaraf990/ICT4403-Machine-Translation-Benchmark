#!/usr/bin/env python3
"""
src/train_mt5.py
Foundations of Generative AI (ICT4403) - Assignment 3
Fine-tunes mT5 (Multilingual T5 - google/mt5-small) for English-to-French Translation.
Evaluates training loss, parameters count, fine-tuning time, and provides inference functions.
"""

import os
import sys
import time
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup


class MT5TranslationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, prefix: str = "translate English to French: ", max_len: int = 64, max_samples: int = None):
        self.data = pd.read_json(data_path, lines=True)
        if max_samples and len(self.data) > max_samples:
            self.data = self.data.iloc[:max_samples]
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = self.prefix + str(row["en"])
        tgt_text = str(row["fr"])

        src_enc = self.tokenizer(
            src_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = tgt_enc["input_ids"].squeeze(0)
        # Hugging Face convention: replace padding token id with -100 to ignore in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": src_enc["input_ids"].squeeze(0),
            "attention_mask": src_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def fine_tune_mt5(
    model_name: str = "google/mt5-small",
    data_dir: str = "data",
    output_dir: str = "results/mt5_finetuned",
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 3e-4,
    train_samples: int = 250,
    val_samples: int = 50,
):
    os.makedirs(output_dir, exist_ok=True)
    # MPS backend on Apple Silicon has numerical instability/NaN bugs with mT5 RMSNorm; use CPU for stable gradients
    device = torch.device("cpu")
    print(f"Fine-tuning mT5 ({model_name}) on device: {device}")

    # 1. Load Tokenizer & Model
    print(f"Loading {model_name} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"mT5 Total Parameters:     {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"mT5 Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # 2. Datasets
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")

    train_ds = MT5TranslationDataset(train_path, tokenizer, max_samples=train_samples)
    val_ds = MT5TranslationDataset(val_path, tokenizer, max_samples=val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Optimizer (Adafactor is the standard, numerically stable optimizer for T5/mT5)
    from transformers import Adafactor
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=5e-4)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # 4. Training loop
    start_time = time.time()
    history = []

    print("-" * 60)
    print(f"Starting Fine-tuning ({epochs} Epochs, {len(train_ds)} samples)...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                print(f"  [Epoch {epoch} | Step {step+1}/{len(train_loader)}] Current Batch Loss: {loss.item():.4f}", flush=True)

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_dur = time.time() - epoch_start
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "duration_sec": epoch_dur,
        })
        print(f"Epoch {epoch:02d} | Avg Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_dur:.1f}s", flush=True)

    total_time = time.time() - start_time
    print(f"\nmT5 Fine-tuning completed in {total_time:.2f}s ({total_time/60:.2f} mins).")

    # 5. Save model and tokenizer
    print(f"Saving fine-tuned mT5 checkpoint to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "fine_tuning_time_sec": total_time,
        "history": history,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    with open(os.path.join(output_dir, "mt5_training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model, tokenizer, total_time, total_params


def translate_mt5(
    model,
    tokenizer,
    sentences,
    device,
    prefix: str = "translate English to French: ",
    max_length: int = 64,
    num_beams: int = 4,
):
    """Batch translation inference using mT5."""
    model.eval()
    if isinstance(sentences, str):
        sentences = [sentences]

    prompts = [prefix + s for s in sentences]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in outputs]


if __name__ == "__main__":
    fine_tune_mt5()
