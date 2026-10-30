#!/usr/bin/env python3
"""
src/temperature_analysis.py
Foundations of Generative AI (ICT4403) - Assignment 3
Systematic Sampling Temperature Sensitivity Analysis for Autoregressive Decoder Models.
Evaluates temperatures T in [0.0, 0.3, 0.7, 1.0, 1.2] measuring BLEU, chrF, lexical diversity,
and hallucination/degeneration rates.
"""

import os
import sys
import time
import json
import torch
import sacrebleu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_llama import LLaMATranslator


def compute_distinct_ngrams(texts: list, n: int = 1) -> float:
    """Compute Distinct-N ratio: unique n-grams divided by total n-grams."""
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        if len(words) >= n:
            ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def run_temperature_experiments(
    data_path: str = "data/test.jsonl",
    output_dir: str = "results",
    num_samples: int = 10,
    temperatures: list = [0.0, 0.3, 0.7, 1.0, 1.2],
):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("SAMPLING TEMPERATURE SENSITIVITY BENCHMARK (DECODER-ONLY GENERATION)")
    print("=" * 70)

    test_df = pd.read_json(data_path, lines=True).iloc[:num_samples]
    sources = test_df["en"].tolist()
    references = test_df["fr"].tolist()

    translator = LLaMATranslator()

    results = []
    temp_translations = {"Source": sources, "Reference": references}

    for temp in temperatures:
        print(f"\nEvaluating Generation at Temperature T = {temp:.1f}...")
        t0 = time.time()
        preds = []
        latencies = []

        for s in sources:
            res = translator.translate(s, temperature=temp)
            preds.append(res["translation"])
            latencies.append(res["latency_sec"])

        elapsed = time.time() - t0
        bleu = sacrebleu.corpus_bleu(preds, [references]).score
        chrf = sacrebleu.corpus_chrf(preds, [references]).score
        distinct_1 = compute_distinct_ngrams(preds, n=1)
        distinct_2 = compute_distinct_ngrams(preds, n=2)
        avg_lat = np.mean(latencies) * 1000

        temp_translations[f"T={temp:.1f}"] = preds

        results.append({
            "Temperature": temp,
            "Sampling Mode": "Greedy (Argmax)" if temp == 0.0 else f"Nucleus Top-p (p=0.9, T={temp:.1f})",
            "BLEU Score": round(bleu, 2),
            "chrF Score": round(chrf, 2),
            "Distinct-1 (Diversity)": round(distinct_1, 4),
            "Distinct-2 (Diversity)": round(distinct_2, 4),
            "Avg Latency (ms)": round(avg_lat, 1),
        })

        print(f"  T={temp:.1f} -> BLEU: {bleu:.2f} | chrF: {chrf:.2f} | Distinct-1: {distinct_1:.3f} | Latency: {avg_lat:.1f}ms")

    # Export CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "temperature_analysis.csv"), index=False)
    pd.DataFrame(temp_translations).to_csv(os.path.join(output_dir, "temperature_translations_samples.csv"), index=False)

    # Plot Temperature Curve
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(9, 5))

    temps = results_df["Temperature"]
    bleus = results_df["BLEU Score"]
    chrfs = results_df["chrF Score"]
    dist2 = results_df["Distinct-2 (Diversity)"] * 100

    color1 = "#2563EB"
    ax1.set_xlabel("Sampling Temperature (T)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Translation Quality (BLEU & chrF)", color=color1, fontsize=12, fontweight="bold")
    l1 = ax1.plot(temps, bleus, "o-", color="#2563EB", linewidth=2.5, markersize=8, label="BLEU Score")
    l2 = ax1.plot(temps, chrfs, "s--", color="#10B981", linewidth=2.5, markersize=8, label="chrF Score")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(max(chrfs) + 10, 50))

    # Twin axis for lexical diversity
    ax2 = ax1.twinx()
    color2 = "#F59E0B"
    ax2.set_ylabel("Lexical Diversity (Distinct-2 %)", color=color2, fontsize=12, fontweight="bold")
    l3 = ax2.plot(temps, dist2, "^-.", color="#F59E0B", linewidth=2.5, markersize=8, label="Distinct-2 (%)")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, fontsize=10)

    plt.title("Effect of Sampling Temperature on Translation Accuracy vs. Diversity", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "temperature_sensitivity.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"\nSaved temperature curve plot to {plot_file}")

    print("\nTemperature Analysis Summary Table:")
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    run_temperature_experiments()
