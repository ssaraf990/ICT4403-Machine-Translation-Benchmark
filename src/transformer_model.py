#!/usr/bin/env python3
"""
src/transformer_model.py
Foundations of Generative AI (ICT4403) - Assignment 3
Basic Transformer (Encoder-Decoder) Architecture built from scratch in PyTorch.
Implements Scaled Dot-Product Attention, Multi-Head Attention, Sinusoidal Positional Encoding,
Encoder Layers, Decoder Layers with Cross-Attention, and Autoregressive Decoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import Optional, Tuple, List, Dict


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding as introduced in 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)

        # 1) Linear projections and split into n_heads
        # [batch_size, seq_len, n_heads, d_k] -> transpose to [batch_size, n_heads, seq_len, d_k]
        q_proj = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2) Scaled Dot-Product Attention: Scores = (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Mask out invalid positions with large negative number
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 3) Multiply weights by Values
        out = torch.matmul(attn_weights, v_proj)  # [batch_size, n_heads, seq_len, d_k]

        # 4) Concatenate heads and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.w_o(out)

        return out, attn_weights


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network with activation and dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN / Post-LN architecture with residual connections
        attn_out, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_out))
        ff_out = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_out))
        return src


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer with Masked Self-Attention and Cross-Attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Masked Self-Attention (causal look-ahead mask)
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_out))

        # 2. Cross-Attention (queries from decoder, keys & values from encoder memory)
        cross_attn_out, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_out))

        # 3. Position-wise Feed Forward
        ff_out = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_out))

        return tgt


class Seq2SeqTransformer(nn.Module):
    """
    Complete Encoder-Decoder Transformer Architecture for Sequence-to-Sequence Translation.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # src: [batch_size, src_len] -> [batch_size, 1, 1, src_len]
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        # tgt: [batch_size, tgt_len]
        batch_size, tgt_len = tgt.size()
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
        # Causal mask: lower triangular matrix of shape [1, 1, tgt_len, tgt_len]
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)
        return pad_mask & subsequent_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        out = self.src_embedding(src) * math.sqrt(self.d_model)
        out = self.pos_encoder(out)
        for layer in self.encoder_layers:
            out = layer(out, src_mask)
        return out

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        out = self.pos_encoder(out)
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_mask, memory_mask)
        return self.fc_out(out)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, src_mask)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TranslationVocab:
    """Vocabulary builder and numericalizer with standard special tokens."""

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, max_vocab: int = 12000, min_freq: int = 2):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.w2i = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.i2w = {idx: word for word, idx in self.w2i.items()}

    def build_vocab(self, texts: List[str]):
        counts = Counter()
        for text in texts:
            words = text.lower().strip().split()
            counts.update(words)

        for word, count in counts.most_common():
            if count < self.min_freq:
                break
            if len(self.w2i) >= self.max_vocab:
                break
            if word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = text.lower().strip().split()
        ids = [self.w2i.get(t, self.w2i[self.UNK_TOKEN]) for t in tokens]
        if add_special_tokens:
            ids = [self.w2i[self.SOS_TOKEN]] + ids + [self.w2i[self.EOS_TOKEN]]
        return ids

    def decode(self, token_ids: List[int], strip_special: bool = True) -> str:
        words = []
        for tid in token_ids:
            if strip_special and tid in [0, 1, 2]:
                if tid == 2:  # Stop at EOS
                    break
                continue
            words.append(self.i2w.get(tid, self.UNK_TOKEN))
        return " ".join(words)

    def __len__(self):
        return len(self.w2i)
