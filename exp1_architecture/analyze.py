"""
Analysis script for Experiment 1 results.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_latest_results():
    results_dir = Path(__file__).parent / 'results'
    result_dirs = sorted(results_dir.glob('*'))
    
    if not result_dirs:
        print("No results found. Run experiment first.")
        return
    
    latest = result_dirs[-1]
    print(f"Analyzing: {latest}")
    
    df = pd.read_csv(latest / 'results.csv')
    
    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ASR by architecture
    asr_by_arch = df.groupby('architecture')['asr_05'].mean().sort_values(ascending=False)
    axes[0].bar(asr_by_arch.index, asr_by_arch.values)
    axes[0].set_title('ASR@0.5 by Architecture')
    axes[0].set_ylabel('ASR@0.5')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Cosine similarity
    cos_by_arch = df.groupby('architecture')['cos_sim_mean'].mean().sort_values(ascending=False)
    axes[1].bar(cos_by_arch.index, cos_by_arch.values)
    axes[1].set_title('Cosine Similarity by Architecture')
    axes[1].set_ylabel('Cos-Sim')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Parameters vs ASR
    for arch in df['architecture'].unique():
        arch_df = df[df['architecture'] == arch]
        axes[2].scatter(arch_df['num_params'].iloc[0], arch_df['asr_05'].mean(), label=arch, s=100)
    axes[2].set_title('Parameters vs ASR')
    axes[2].set_xlabel('Parameters')
    axes[2].set_ylabel('ASR@0.5')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(latest / 'analysis.png', dpi=150)
    plt.show()
    
    print("\nBest architecture:")
    with open(latest / 'best_architecture.json') as f:
        print(json.dumps(json.load(f), indent=2))


if __name__ == '__main__':
    analyze_latest_results()



