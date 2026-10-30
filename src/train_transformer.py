#!/usr/bin/env python3
"""
src/train_transformer.py
Foundations of Generative AI (ICT4403) - Assignment 3
Trains the Basic Transformer (Encoder-Decoder) from scratch on English-French pairs.
Records training time, parameter count, validation losses, and exports the trained model.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_model import Seq2SeqTransformer, TranslationVocab


class TranslationDataset(Dataset):
    def __init__(self, data_path: str, src_vocab: TranslationVocab, tgt_vocab: TranslationVocab, max_len: int = 50):
        self.data = pd.read_json(data_path, lines=True)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_ids = self.src_vocab.encode(row["en"], add_special_tokens=True)[: self.max_len]
        tgt_ids = self.tgt_vocab.encode(row["fr"], add_special_tokens=True)[: self.max_len]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_idx=0):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)

    padded_src = torch.full((len(batch), max_src_len), pad_idx, dtype=torch.long)
    padded_tgt = torch.full((len(batch), max_tgt_len), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        padded_src[i, : len(s)] = s
        padded_tgt[i, : len(t)] = t

    return padded_src, padded_tgt


def greedy_decode(
    model: Seq2SeqTransformer,
    src_sentence: str,
    src_vocab: TranslationVocab,
    tgt_vocab: TranslationVocab,
    device: torch.device,
    max_len: int = 50,
) -> str:
    """Generate translation autoregressively via greedy search."""
    model.eval()
    src_ids = src_vocab.encode(src_sentence, add_special_tokens=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        memory = model.encode(src_tensor, src_mask)

    tgt_tokens = [tgt_vocab.w2i[tgt_vocab.SOS_TOKEN]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        tgt_mask = model.make_tgt_mask(tgt_tensor)

        with torch.no_grad():
            output = model.decode(tgt_tensor, memory, tgt_mask, src_mask)
            next_token = output[0, -1, :].argmax().item()

        tgt_tokens.append(next_token)
        if next_token == tgt_vocab.w2i[tgt_vocab.EOS_TOKEN]:
            break

    return tgt_vocab.decode(tgt_tokens, strip_special=True)


def train_basic_transformer(
    data_dir: str = "data",
    output_dir: str = "results",
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 5e-4,
    d_model: int = 256,
    n_heads: int = 8,
    num_layers: int = 3,
    d_ff: int = 512,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Basic Transformer on device: {device}")

    # 1. Load data
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)

    # 2. Build vocabularies
    print("Building English and French vocabularies...")
    src_vocab = TranslationVocab(max_vocab=10000, min_freq=2)
    tgt_vocab = TranslationVocab(max_vocab=10000, min_freq=2)

    src_vocab.build_vocab(train_df["en"].tolist())
    tgt_vocab.build_vocab(train_df["fr"].tolist())
    print(f"Source (En) Vocab Size: {len(src_vocab):,}")
    print(f"Target (Fr) Vocab Size: {len(tgt_vocab):,}")

    train_dataset = TranslationDataset(train_path, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_path, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Initialize Model
    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1,
        pad_idx=0,
    ).to(device)

    total_params = model.count_parameters()
    print(f"Basic Transformer Total Trainable Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # 4. Training Loop
    start_time = time.time()
    history = []

    print("-" * 60)
    print(f"Starting Training for {epochs} Epochs...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            # Teacher forcing: feed tgt[:, :-1], predict tgt[:, 1:]
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)  # [batch, seq_len, vocab_size]

            loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_target = tgt[:, 1:]
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_duration = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "duration_sec": epoch_duration,
        })

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Time: {epoch_duration:.1f}s"
        )

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes).")

    # 5. Save model checkpoint & artifacts
    checkpoint_path = os.path.join(output_dir, "basic_transformer.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "d_model": d_model,
                "n_heads": n_heads,
                "num_layers": num_layers,
                "d_ff": d_ff,
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
            },
            "history": history,
            "total_params": total_params,
            "training_time_sec": total_training_time,
        },
        checkpoint_path,
    )

    vocab_path = os.path.join(output_dir, "transformer_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "src_w2i": src_vocab.w2i,
                "tgt_w2i": tgt_vocab.w2i,
            },
            f,
            ensure_ascii=False,
        )

    # Test sample translation
    sample_src = "The European Commission submitted a new proposal."
    sample_trans = greedy_decode(model, sample_src, src_vocab, tgt_vocab, device)
    print(f"\nSample Inference Test:")
    print(f"  Input:       {sample_src}")
    print(f"  Translation: {sample_trans}")

    return model, src_vocab, tgt_vocab, total_training_time, total_params


if __name__ == "__main__":
    train_basic_transformer()
