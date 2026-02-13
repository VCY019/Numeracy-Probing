#!/usr/bin/env python3
"""
Probe Training Script for Arxiv Numerical Data

This script trains regression probes on embeddings extracted from arxiv numerical datasets.
Trains probes for decimal, scientific, and mixed (decimal+scientific) notation.

Usage:
    python train_probe_arxiv.py --data_path_decimal <path> --data_path_scientific <path> --model_name <str> --num_layers <int>
"""

import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_metrics(y_true, y_pred, log_space=True):
    """Calculate evaluation metrics including relative error."""
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    mse = mean_squared_error(y_true, y_pred)
    
    if log_space:
        relative_error = np.abs(1 - np.exp2(y_pred - y_true))
        mean_relative_error = np.mean(relative_error)
        
        y_true_orig = np.exp2(y_true)
        y_pred_orig = np.exp2(y_pred)
        tolerance = 0.01 * y_true_orig
        aacc = np.mean(np.abs(y_pred_orig - y_true_orig) <= tolerance)
    else:
        relative_error = np.abs((y_pred - y_true) / y_true)
        mean_relative_error = np.mean(relative_error)
        
        tolerance = 0.01 * y_true
        aacc = np.mean(np.abs(y_pred - y_true) <= tolerance)
    
    return {
        'r2': r2,
        'pearson': pearson, 
        'mse': mse,
        'aacc': aacc,
        'mean_relative_error': mean_relative_error
    }


def generate_regression_plots(X_test_all, y_test, models, offset_type, results_dir, num_layers, best_layer_idx):
    """Generate regression scatter plots: first, best (by val R2), last."""
    sns.set_style("darkgrid")
    sns.set_context(rc={"axes.labelsize": 16, "legend.fontsize": 16, "legend.title_fontsize": 16})

    plot_configs = [
        (0, "Layer 1", f"{offset_type}_first_layer{1}_scatter.pdf"),
        (best_layer_idx, f"Layer {best_layer_idx + 1}", f"{offset_type}_best_layer{best_layer_idx + 1}_scatter.pdf"),
        (num_layers-1, f"Layer {num_layers}", f"{offset_type}_last_layer{num_layers}_scatter.pdf")
    ]
    
    for layer_idx, title, filename in plot_configs:
        if models[layer_idx] is not None:
            y_pred = models[layer_idx].predict(X_test_all[layer_idx])
            
            df = pd.DataFrame({
                'log(Prediction)': y_pred,
                'log(Golden)': y_test,
            })
            
            grid = sns.lmplot(data=df, x='log(Golden)', y='log(Prediction)', 
                            fit_reg=False, scatter_kws={'s': 1}, height=3.3, aspect=1)
            ax = grid.ax
            ax.set_aspect('equal', adjustable='box')
            
            if y_test.min() < -200 or y_test.max() > 200:
                plt.plot([-200, 200], [-200, 200], color='#e994a5', linewidth=2)
                plt.xlim(-200, 200)
                plt.ylim(-200, 200)
            else:
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                        color='#e994a5', linewidth=2)
            
            plt.tight_layout()
            
            plot_path = os.path.join(results_dir, filename)
            grid.figure.savefig(plot_path, dpi=150)
            grid.figure.savefig(plot_path.replace('.pdf', '.png'), dpi=150)
            plt.close()
    
    logger.info(f"Generated scatter plots for {offset_type} in {results_dir}")


def train_probe(embeddings_offset0, embeddings_offset1, values, name, args):
    """
    Train regression probes for both offset types.
    
    Args:
        embeddings_offset0: List of embeddings for offset_0 [layer_1, layer_2, ..., layer_n]
        embeddings_offset1: List of embeddings for offset_1 [layer_1, layer_2, ..., layer_n]
        values: Array of numeric values (not log-transformed)
        name: Name for saving results (e.g., 'decimal', 'scientific', 'mixed')
        args: Command line arguments
    """
    log_values = np.log2(values)
    logger.info(f"Training probes for {name}: {len(values)} samples")
    
    # 80/10/10 split
    train_split_idx = int(0.8 * len(values))
    val_split_idx = int(0.9 * len(values))
    
    y_train = log_values[:train_split_idx]
    y_val = log_values[train_split_idx:val_split_idx]
    y_test = log_values[val_split_idx:]
    
    logger.info(f"Split: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    
    # Create output directories
    results_dir = os.path.join('results_arxiv', args.model_name, name)
    models_dir = os.path.join('probes_arxiv', args.model_name, name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Train for both offset types
    for offset_type, embeddings_all in [('offset_0', embeddings_offset0), ('offset_1', embeddings_offset1)]:
        logger.info(f"Training {offset_type} probes for {name}...")
        
        offset_results = []
        offset_models = []
        
        # Split embeddings
        X_train_all = [emb[:train_split_idx] for emb in embeddings_all]
        X_val_all = [emb[train_split_idx:val_split_idx] for emb in embeddings_all]
        X_test_all = [emb[val_split_idx:] for emb in embeddings_all]
        
        # Train each layer
        for layer_idx in tqdm(range(args.num_layers), desc=f"Training {offset_type}"):
            X_train = X_train_all[layer_idx]
            X_val = X_val_all[layer_idx]
            X_test = X_test_all[layer_idx]
            
            model_path = os.path.join(models_dir, f'{offset_type}_layer_{layer_idx+1}.pkl')
            
            if args.load_model and os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded model for {offset_type} layer {layer_idx+1}")
            else:
                model = Ridge(alpha=0.1)
                model.fit(X_train, y_train)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Evaluate
            y_val_pred = model.predict(X_val)
            val_metrics = evaluate_metrics(y_val, y_val_pred, log_space=True)

            y_test_pred = model.predict(X_test)
            test_metrics = evaluate_metrics(y_test, y_test_pred, log_space=True)

            metrics = {
                'layer': layer_idx + 1,
                'val_r2': val_metrics['r2'],
                'val_pearson': val_metrics['pearson'],
                'val_mse': val_metrics['mse'],
                'val_aacc': val_metrics['aacc'],
                'val_mean_relative_error': val_metrics['mean_relative_error'],
                'test_r2': test_metrics['r2'],
                'test_pearson': test_metrics['pearson'],
                'test_mse': test_metrics['mse'],
                'test_aacc': test_metrics['aacc'],
                'test_mean_relative_error': test_metrics['mean_relative_error'],
            }
            
            offset_results.append(metrics)
            offset_models.append(model)
        
        # Save results
        results_path = os.path.join(results_dir, f'{offset_type}_results.json')
        with open(results_path, 'w') as f:
            json.dump(offset_results, f, indent=2)
        logger.info(f"Saved {offset_type} results to {results_path}")
        
        # Find best layer by validation R2
        best_layer = max(offset_results, key=lambda x: x['val_r2'])
        best_layer_idx = best_layer['layer'] - 1

        logger.info(f"{offset_type} best layer {best_layer['layer']} (selected by val R2):")
        logger.info(f"  Val  - R2={best_layer['val_r2']:.4f}, Pearson={best_layer['val_pearson']:.4f}, "
                   f"AACC={best_layer['val_aacc']:.4f}, Rel Error={best_layer['val_mean_relative_error']:.4f}")
        logger.info(f"  Test - R2={best_layer['test_r2']:.4f}, Pearson={best_layer['test_pearson']:.4f}, "
                   f"AACC={best_layer['test_aacc']:.4f}, Rel Error={best_layer['test_mean_relative_error']:.4f}")
        
        # Generate plots
        generate_regression_plots(X_test_all, y_test, offset_models, offset_type, results_dir, args.num_layers, best_layer_idx)


def load_embeddings(data_path, offset_type, valid_indices, num_layers):
    """Load embeddings for all layers."""
    embeddings_all = []
    for layer_idx in range(num_layers):
        embed_path = os.path.join(data_path, offset_type, f'layer_{layer_idx+1}.embeds')
        embeddings = []
        with open(embed_path, 'rb') as f:
            while True:
                try:
                    embeddings.append(np.load(f))
                except EOFError:
                    break
        embeddings = np.array(embeddings)[valid_indices]
        embeddings_all.append(embeddings)
    return embeddings_all


def extract_values(metadata):
    """Extract numeric values from metadata and return valid indices."""
    values = []
    valid_indices = []
    for i, meta in enumerate(metadata):
        if 'numeric_value' in meta and meta['numeric_value'] is not None and meta['numeric_value'] > 0:
            values.append(meta['numeric_value'])
            valid_indices.append(i)
    return np.array(values), valid_indices


def main():
    parser = argparse.ArgumentParser(description="Train regression probes on arxiv numerical embeddings")
    parser.add_argument("--data_path_decimal", type=str, required=True, help="Path to decimal embeddings directory")
    parser.add_argument("--data_path_scientific", type=str, required=True, help="Path to scientific embeddings directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for output directories")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of model layers")
    parser.add_argument("--load_model", action="store_true", help="Load existing models instead of training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    np.random.seed(args.seed)
    
    # Load decimal data
    logger.info("Loading decimal data...")
    with open(os.path.join(args.data_path_decimal, 'metadata.jsonl'), 'r') as f:
        metadata_dec = [json.loads(line) for line in f]
    values_dec, valid_indices_dec = extract_values(metadata_dec)
    logger.info(f"Decimal: {len(values_dec)} valid samples")
    
    embeddings_dec_offset0 = load_embeddings(args.data_path_decimal, 'offset_0', valid_indices_dec, args.num_layers)
    embeddings_dec_offset1 = load_embeddings(args.data_path_decimal, 'offset_1', valid_indices_dec, args.num_layers)
    
    # Load scientific data
    logger.info("Loading scientific data...")
    with open(os.path.join(args.data_path_scientific, 'metadata.jsonl'), 'r') as f:
        metadata_sci = [json.loads(line) for line in f]
    values_sci, valid_indices_sci = extract_values(metadata_sci)
    logger.info(f"Scientific: {len(values_sci)} valid samples")
    
    embeddings_sci_offset0 = load_embeddings(args.data_path_scientific, 'offset_0', valid_indices_sci, args.num_layers)
    embeddings_sci_offset1 = load_embeddings(args.data_path_scientific, 'offset_1', valid_indices_sci, args.num_layers)
    
    # Prepare mixed data (interleave decimal and scientific)
    logger.info("Preparing mixed data...")
    n_dec = len(values_dec)
    n_sci = len(values_sci)
    n_min = min(n_dec, n_sci)
    
    # Interleave: [dec[0], sci[0], dec[1], sci[1], ...]
    embeddings_mixed_offset0 = []
    embeddings_mixed_offset1 = []
    for layer_idx in range(args.num_layers):
        interleaved_offset0 = []
        interleaved_offset1 = []
        for i in range(n_min):
            interleaved_offset0.append(embeddings_dec_offset0[layer_idx][i])
            interleaved_offset0.append(embeddings_sci_offset0[layer_idx][i])
            interleaved_offset1.append(embeddings_dec_offset1[layer_idx][i])
            interleaved_offset1.append(embeddings_sci_offset1[layer_idx][i])
        # Add remaining samples if counts differ
        if n_dec > n_min:
            interleaved_offset0.extend(embeddings_dec_offset0[layer_idx][n_min:])
            interleaved_offset1.extend(embeddings_dec_offset1[layer_idx][n_min:])
        elif n_sci > n_min:
            interleaved_offset0.extend(embeddings_sci_offset0[layer_idx][n_min:])
            interleaved_offset1.extend(embeddings_sci_offset1[layer_idx][n_min:])
        
        embeddings_mixed_offset0.append(np.array(interleaved_offset0))
        embeddings_mixed_offset1.append(np.array(interleaved_offset1))
    
    # Interleave values
    values_mixed = []
    for i in range(n_min):
        values_mixed.append(values_dec[i])
        values_mixed.append(values_sci[i])
    if n_dec > n_min:
        values_mixed.extend(values_dec[n_min:])
    elif n_sci > n_min:
        values_mixed.extend(values_sci[n_min:])
    values_mixed = np.array(values_mixed)
    
    logger.info(f"Mixed: {len(values_mixed)} total samples (interleaved)")
    
    # Train decimal probes
    logger.info("=" * 80)
    logger.info("TRAINING PROBES FOR: DECIMAL")
    logger.info("=" * 80)
    train_probe(embeddings_dec_offset0, embeddings_dec_offset1, values_dec, 'decimal', args)
    
    # Train scientific probes
    logger.info("=" * 80)
    logger.info("TRAINING PROBES FOR: SCIENTIFIC")
    logger.info("=" * 80)
    train_probe(embeddings_sci_offset0, embeddings_sci_offset1, values_sci, 'scientific', args)
    
    # Train mixed probes
    logger.info("=" * 80)
    logger.info("TRAINING PROBES FOR: MIXED")
    logger.info("=" * 80)
    train_probe(embeddings_mixed_offset0, embeddings_mixed_offset1, values_mixed, 'mixed', args)
    
    logger.info("=" * 80)
    logger.info("ALL TRAINING COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()