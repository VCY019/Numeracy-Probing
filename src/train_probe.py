#!/usr/bin/env python3
"""
Probe Training Script for Numerical Comparison Datasets

Trains linear probes (Ridge regression and logistic regression) on train split 
embeddings with validation evaluation, and optionally evaluates on test split 
with --eval_test.

Supports three probe types:
  - regression: predicts log2(value) for individual numbers
  - classification: predicts whether a > b for number pairs
  - regression_diff: predicts log2(a/b) = log2(a) - log2(b) for pairs

Regression probes generate scatter plots (true vs predicted) for first, best 
(selected by val R²), and last layers. Plots are only created with --eval_test.

For mixed-notation datasets (e.g., int+sci), --cross_notation_eval evaluates 
regression probes in both cross-notation directions (e.g., train on int → test 
on sci, and train on sci → test on int).

Outputs:
    results/{model}/{dataset}/{probe_type}_probes.pkl
    results/{model}/{dataset}/{probe_type}/{probe_name}/val_results.json
    results/{model}/{dataset}/{probe_type}/{probe_name}/test_results.json (if --eval_test)
    results/{model}/{dataset}/{probe_type}/{probe_name}/test_preds.npy (if --eval_test)
    results/{model}/{dataset}/{probe_type}/{probe_name}/*.{pdf,png} (if --eval_test)
    results/{model}/{dataset}/{probe_type}/cross_{X}_to_{Y}/{probe_name}/* (if --cross_notation_eval)

Usage:
    # Train all probes and save validation results
    python train_probe.py --data_dir data/int_sci_compare/ --embed_dir embeddings/Mistral-7B/int_sci_compare/ --num_layers 32 --model_name Mistral-7B

    # Evaluate on test set with scatter plots
    python train_probe.py --data_dir data/int_sci_compare/ --embed_dir embeddings/Mistral-7B/int_sci_compare/ --num_layers 32 --model_name Mistral-7B --eval_test

    # Load pre-trained probes and evaluate on test
    python train_probe.py --data_dir data/int_sci_compare/ --embed_dir embeddings/Mistral-7B/int_sci_compare/ --num_layers 32 --model_name Mistral-7B --load_probes --eval_test

    # Cross-notation evaluation (requires --eval_test and mixed-notation dataset)
    python train_probe.py --data_dir data/int_sci_compare/ --embed_dir embeddings/Mistral-7B/int_sci_compare/ --num_layers 32 --model_name Mistral-7B --eval_test --cross_notation_eval
    
    # Train only log-ratio regression probes
    python train_probe.py --data_dir data/int_sci_compare/ --embed_dir embeddings/Mistral-7B/int_sci_compare/ --num_layers 32 --model_name Mistral-7B --probe_types regression_diff
"""

import os
import argparse
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import logging
from tqdm import tqdm
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProbeTrainer:
    def __init__(self, data_dir, embed_dir, num_layers, model_name, eval_test, cross_notation_eval=None, seed=42):
        self.data_dir = data_dir
        self.embed_dir = embed_dir
        self.num_layers = num_layers
        self.model_name = model_name
        self.eval_test = eval_test
        self.seed = seed
        self.dataset_name = os.path.basename(data_dir.rstrip('/'))
        self.cross_notation_eval = cross_notation_eval
        self._prepared_data_cache = {}

        np.random.seed(seed)

        # Load data and embeddings for all splits
        self.data = {}
        self.embeddings = {}
        for split in ['train', 'val', 'test']:
            self.data[split] = self._load_data_split(split)
            self.embeddings[split] = self._load_embeddings_split(split)

        # Determine notation types from data
        self.notations = self._get_notations()
        self.is_mixed_notation = len(self.notations) > 1

        if cross_notation_eval:
            if not self.is_mixed_notation:
                raise ValueError("cross_notation_eval can only be used with mixed notation datasets")

        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Notations: {self.notations}")
        logger.info(f"Mixed notation: {self.is_mixed_notation}")

    def _load_data_split(self, split):
        """Load data for a specific split."""

        path = os.path.join(self.data_dir, f'{split}.jsonl')
        if not os.path.exists(path):
            return None

        samples = []
        with open(path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                try:
                    sample['val_a'] = float(eval(sample['a'].replace('×', '*').replace('^', '**')))
                    sample['val_b'] = float(eval(sample['b'].replace('×', '*').replace('^', '**')))
                    samples.append(sample)
                except:
                    logger.info(f"Error loading sample: {sample}")
                    continue

        logger.info(f"Loaded {len(samples)} {split} samples")
        return samples

    def _load_embeddings_split(self, split):
        """Load embeddings for a specific split."""

        embeddings = {}
        split_dir = os.path.join(self.embed_dir, split)

        if not os.path.exists(split_dir):
            return None

        for embed_type in ['offset_0', 'offset_1', 'last_token']:
            embed_path = os.path.join(split_dir, embed_type)
            if not os.path.exists(embed_path):
                continue

            layers = []
            for layer_idx in range(1, self.num_layers + 1):
                filepath = os.path.join(embed_path, f'layer_{layer_idx}.embeds')
                layer_data = []
                try:
                    with open(filepath, 'rb') as f:
                        while True:
                            layer_data.append(np.load(f))
                except EOFError:
                    pass
                layers.append(np.array(layer_data))

            embeddings[embed_type] = np.array(layers) # Shape: (layers, samples, features)
            logger.info(f"Loaded {split} {embed_type} embeddings: shape {embeddings[embed_type].shape}")

        return embeddings

    def _get_notations(self):
        """Extract unique notations from training data."""
        notations = set()
        if self.data['train']:
            for sample in self.data['train']:
                notations.update(sample['notations'])
        return sorted(list(notations))

    def prepare_regression_data(self, split):
        """Prepare regression data for a split."""
        cache_key = f"regression_{split}"
        if cache_key in self._prepared_data_cache:
            logger.info(f"Using cached regression data for {split}")
            return self._prepared_data_cache[cache_key]

        data_dict = {}

        if not self.data[split] or not self.embeddings[split]:
            return data_dict

        # Extract log values for all numbers
        y = []
        notations_list = []
        for sample in self.data[split]:
            y.extend([np.log2(sample['val_a']), np.log2(sample['val_b'])])
            notations_list.extend(sample['notations'])

        y = np.array(y)
        notations_array = np.array(notations_list)

        # Prepare different probe types based on notation
        for offset_type in ['offset_0', 'offset_1']:
            if offset_type not in self.embeddings[split]:
                continue

            X = self.embeddings[split][offset_type]

            if self.is_mixed_notation:
                # For mixed notation, create separate probes for each notation and mixed
                for notation in self.notations:
                    indices = np.where(notations_array == notation)[0]
                    probe_name = f"{notation}_{offset_type}"
                    data_dict[probe_name] = {
                        'X': X[:, indices, :],
                        'y': y[indices]
                    }

                # Mixed probe with all data
                data_dict[f"mixed_{offset_type}"] = {
                    'X': X,
                    'y': y
                }
            else:
                # Single notation dataset
                data_dict[offset_type] = {
                    'X': X,
                    'y': y
                }

        self._prepared_data_cache[cache_key] = data_dict
        return data_dict

    def prepare_regression_diff_data(self, split):
        """Prepare log-ratio regression data (log2(a) - log2(b)) for a split."""
        cache_key = f"regression_diff_{split}"
        if cache_key in self._prepared_data_cache:
            logger.info(f"Using cached regression difference data for {split}")
            return self._prepared_data_cache[cache_key]

        data_dict = {}

        if not self.data[split] or not self.embeddings[split]:
            return data_dict

        # Extract the log difference of a and b
        y = np.array([np.log2(sample['val_a']) - np.log2(sample['val_b']) for sample in self.data[split]])

        # Last token probe
        if 'last_token' in self.embeddings[split]:
            data_dict['last_token'] = {
                'X': self.embeddings[split]['last_token'],
                'y': y
            }

        # Concatenated offset probes
        for offset_type in ['offset_0', 'offset_1']:
            if offset_type not in self.embeddings[split]:
                continue

            # Split embeddings for a and b (they're interleaved)
            embeds = self.embeddings[split][offset_type]
            embeds_a = embeds[:, ::2, :]
            embeds_b = embeds[:, 1::2, :]

            probe_name = f"concat_{offset_type}"
            data_dict[probe_name] = {
                'X': np.concatenate([embeds_a, embeds_b], axis=2),
                'y': y
            }

        self._prepared_data_cache[cache_key] = data_dict
        return data_dict

    def prepare_classification_data(self, split):
        """Prepare classification data for a split."""
        cache_key = f"classification_{split}"
        if cache_key in self._prepared_data_cache:
            logger.info(f"Using cached classification data for {split}")
            return self._prepared_data_cache[cache_key]

        data_dict = {}

        if not self.data[split] or not self.embeddings[split]:
            return data_dict

        # Extract comparison labels
        y = np.array([sample['val_a'] > sample['val_b'] for sample in self.data[split]])

        # Last token probe
        if 'last_token' in self.embeddings[split]:
            data_dict['last_token'] = {
                'X': self.embeddings[split]['last_token'],
                'y': y
            }

        # Concatenated offset probes
        for offset_type in ['offset_0', 'offset_1']:
            if offset_type not in self.embeddings[split]:
                continue

            # Split embeddings for a and b (they're interleaved)
            embeds = self.embeddings[split][offset_type]
            embeds_a = embeds[:, ::2, :]
            embeds_b = embeds[:, 1::2, :]

            probe_name = f"concat_{offset_type}"
            data_dict[probe_name] = {
                'X': np.concatenate([embeds_a, embeds_b], axis=2),
                'y': y
            }

        self._prepared_data_cache[cache_key] = data_dict
        return data_dict

    def train_regression_probes(self, train_data, val_data, probe_type):
        """Train Ridge regression probes for all layers on training data, evaluate on validation."""
        probes = {}
        results = {}

        for probe_name in train_data:
            logger.info(f"Training regression probe: {probe_name}")

            X_train = train_data[probe_name]['X']
            y_train = train_data[probe_name]['y']
            X_val = val_data[probe_name]['X']
            y_val = val_data[probe_name]['y']

            models = []
            probe_results = []

            for layer_idx in tqdm(range(self.num_layers), desc=probe_name, leave=False):
                model = Ridge(alpha=1.0)
                model.fit(X_train[layer_idx], y_train)
                models.append(model)

                # Evaluate on validation
                y_pred = model.predict(X_val[layer_idx])
                if probe_type == 'regression':
                    probe_results.append({
                        'train_r2': model.score(X_train[layer_idx], y_train),
                        'val_r2': model.score(X_val[layer_idx], y_val),
                        'val_mse': mean_squared_error(y_val, y_pred),
                        'val_pearson': pearsonr(y_val, y_pred)[0],
                        'val_acc_1%': np.mean(np.abs(np.exp2(y_val) - np.exp2(y_pred)) <= 0.01 * np.abs(np.exp2(y_val)))
                    })
                elif probe_type == 'regression_diff':
                    probe_results.append({
                        'train_r2': model.score(X_train[layer_idx], y_train),
                        'val_r2': model.score(X_val[layer_idx], y_val),
                        'val_mse': mean_squared_error(y_val, y_pred),
                        'val_pearson': pearsonr(y_val, y_pred)[0],
                        'val_acc': np.mean(np.sign(y_pred) == np.sign(y_val))
                    })

            probes[probe_name] = models
            results[probe_name] = probe_results

        return probes, results

    def train_classification_probes(self, train_data, val_data):
        """Train logistic classification probes for all layers on training data, evaluate on validation."""
        probes = {}
        results = {}

        for probe_name in train_data:
            logger.info(f"Training classification probe: {probe_name}")

            X_train = train_data[probe_name]['X']
            y_train = train_data[probe_name]['y']
            X_val = val_data[probe_name]['X']
            y_val = val_data[probe_name]['y']

            models = []
            probe_results = []

            for layer_idx in tqdm(range(self.num_layers), desc=probe_name, leave=False):
                model = LogisticRegression(max_iter=10_000)
                model.fit(X_train[layer_idx], y_train)
                models.append(model)

                probe_results.append({
                    'train_acc': model.score(X_train[layer_idx], y_train),
                    'val_acc': model.score(X_val[layer_idx], y_val)
                })

            probes[probe_name] = models
            results[probe_name] = probe_results

        return probes, results

    def evaluate_probes(self, probes, test_data, probe_type):
        """Evaluate probes on test data and return metrics and predictions."""
        results = {}
        all_preds = {}

        for probe_name, models in probes.items():
            if probe_name not in test_data:
                continue

            X_test = test_data[probe_name]['X']
            y_test = test_data[probe_name]['y']

            probe_results = []
            preds = []
            for layer_idx, model in enumerate(models):
                score = model.score(X_test[layer_idx], y_test)
                y_pred = model.predict(X_test[layer_idx])
                preds.append(y_pred)
                if probe_type == 'regression':
                    probe_results.append({
                        'test_r2': score,
                        'test_mse': mean_squared_error(y_test, y_pred),
                        'test_pearson': pearsonr(y_test, y_pred)[0],
                        'test_acc_1%': np.mean(np.abs(np.exp2(y_test) - np.exp2(y_pred)) <= 0.01 * np.abs(np.exp2(y_test)))
                    })
                elif probe_type == 'regression_diff':
                    probe_results.append({
                        'test_r2': score,
                        'test_mse': mean_squared_error(y_test, y_pred),
                        'test_pearson': pearsonr(y_test, y_pred)[0],
                        'test_acc': np.mean(np.sign(y_pred) == np.sign(y_test))
                    })
                elif probe_type == 'classification':
                    probe_results.append({'test_acc': score})

            results[probe_name] = probe_results
            all_preds[probe_name] = preds

        return results, all_preds

    def plot_regression_results(self, models, val_data, test_data, probe_name, output_dir, probe_type, subdir=""):
        """
        Generate regression scatter plots on test data.
        Best layer is selected using validation R² score.
        """
        X_val = val_data[probe_name]['X']
        y_val = val_data[probe_name]['y']
        X_test = test_data[probe_name]['X']
        y_test = test_data[probe_name]['y']

        # Find best layer
        best_layer_idx = 0
        best_score = -float('inf')
        for i, model in enumerate(models):
            score = model.score(X_val[i], y_val)
            if score > best_score:
                best_score = score
                best_layer_idx = i

        # Generate plots
        plot_configs = [
            (0, "Layer 1", f"first_layer1_scatter.pdf"),
            (best_layer_idx, f"Layer {best_layer_idx + 1}", f"best_layer{best_layer_idx + 1}_scatter.pdf"),
            (-1, f"Layer {self.num_layers}", f"last_layer{self.num_layers}_scatter.pdf")
        ]

        sns.set_style("darkgrid")
        sns.set_context(rc={"axes.labelsize":16, "legend.fontsize":16, "legend.title_fontsize":16})

        for layer_idx, title, filename in plot_configs:
            y_pred = models[layer_idx].predict(X_test[layer_idx])

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
            if subdir:
                plot_path = os.path.join(output_dir, probe_type, subdir, probe_name, filename)
            else:
                plot_path = os.path.join(output_dir, probe_type, probe_name, filename)
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            grid.figure.savefig(plot_path, dpi=150)
            grid.figure.savefig(plot_path.replace('.pdf', '.png'), dpi=150)
            plt.close()

    def save_probes(self, probes, probe_type, output_dir):
        """Save trained probes."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{probe_type}_probes.pkl'), 'wb') as f:
            pickle.dump(probes, f)
        logger.info(f"Saved {probe_type} probes to {output_dir}")       

    def load_probes(self, output_dir, probe_type):
        """Load existing probes for a given probe type."""
        probe_path = os.path.join(output_dir, f'{probe_type}_probes.pkl')
        with open(probe_path, 'rb') as f:
            probes = pickle.load(f)
        
        logger.info(f"Loaded {probe_type} probes from {probe_path}")
        return probes

    def save_results(self, results, preds, output_dir, probe_type, split, subdir=""):
        """Save results and predictions for a given split."""
        for probe_name, probe_results in results.items():
            if subdir:
                probe_dir = os.path.join(output_dir, probe_type, subdir, probe_name)
            else:
                probe_dir = os.path.join(output_dir, probe_type, probe_name)
            os.makedirs(probe_dir, exist_ok=True)

            with open(os.path.join(probe_dir, f'{split}_results.json'), 'w') as f:
                json.dump(probe_results, f, indent=4)
            
            # Save predictions if provided
            if preds and probe_name in preds:
                np.save(os.path.join(probe_dir, f'{split}_preds.npy'), preds[probe_name])
        

        logger.info(f"Saved {probe_type} results and predictions for {split} to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory with train/val/test.jsonl')
    parser.add_argument('--embed_dir', required=True, help='Directory with train/val/test embeddings')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of model layers')
    parser.add_argument('--model_name', required=True, help='Model name for output paths')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--load_probes', action='store_true', help='Load existing probes instead of training')
    parser.add_argument('--load_probes_dir', default=None, help='Directory to load existing probes from')
    parser.add_argument('--eval_test', action='store_true', help='Evaluate on test set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--probe_types', nargs='+', default=['regression', 'classification'],
                         choices=['regression', 'classification', 'regression_diff'], help='Probe types to train')
    parser.add_argument('--cross_notation_eval', action='store_true',
                    help='For mixed notation datasets, evaluate in both directions (e.g., int→sci and sci→int)')

    args = parser.parse_args()

    # Initialize trainer
    trainer = ProbeTrainer(args.data_dir, args.embed_dir, args.num_layers, args.model_name, args.eval_test, args.cross_notation_eval, args.seed)

    # Set output directory
    output_dir = os.path.join(args.output_dir, args.model_name, trainer.dataset_name)
    # Set load directory (use output_dir if not specified)
    load_dir = os.path.join(args.load_probes_dir or args.output_dir, args.model_name, trainer.dataset_name)

    if args.load_probes:
        # Load existing probes
        if "regression" in args.probe_types:
            reg_probes = trainer.load_probes(load_dir, 'regression')
        if "classification" in args.probe_types:
            cls_probes = trainer.load_probes(load_dir, 'classification')
        if "regression_diff" in args.probe_types:
            reg_diff_probes = trainer.load_probes(load_dir, 'regression_diff')
    else:
        # Train probes
        if "regression" in args.probe_types:
            train_reg_data = trainer.prepare_regression_data('train')
            val_reg_data = trainer.prepare_regression_data('val')

            logger.info("Training regression probes...")
            reg_probes, reg_results = trainer.train_regression_probes(train_reg_data, val_reg_data, 'regression')

            trainer.save_probes(reg_probes, 'regression', output_dir)
            trainer.save_results(reg_results, None, output_dir, 'regression', 'val')

        if "classification" in args.probe_types:
            train_cls_data = trainer.prepare_classification_data('train')
            val_cls_data = trainer.prepare_classification_data('val')

            logger.info("Training classification probes...")
            cls_probes, cls_results = trainer.train_classification_probes(train_cls_data, val_cls_data)
            
            trainer.save_probes(cls_probes, 'classification', output_dir)
            trainer.save_results(cls_results, None, output_dir, 'classification', 'val')

        if "regression_diff" in args.probe_types:
            train_reg_diff_data = trainer.prepare_regression_diff_data('train')
            val_reg_diff_data = trainer.prepare_regression_diff_data('val')

            logger.info("Training regression difference probes...")
            reg_diff_probes, reg_diff_results = trainer.train_regression_probes(train_reg_diff_data, val_reg_diff_data, 'regression_diff')

            trainer.save_probes(reg_diff_probes, 'regression_diff', output_dir)
            trainer.save_results(reg_diff_results, None, output_dir, 'regression_diff', 'val')

    # Evaluate and plot on test set if requested
    if args.eval_test:
        logger.info("Evaluating and plotting on test set...")
        if "regression" in args.probe_types:
            val_reg_data = trainer.prepare_regression_data('val')
            test_reg_data = trainer.prepare_regression_data('test')
            test_reg_results, test_reg_preds = trainer.evaluate_probes(reg_probes, test_reg_data, 'regression')
            trainer.save_results(test_reg_results, test_reg_preds, output_dir, 'regression', 'test')

            for probe_name, models in reg_probes.items():
                trainer.plot_regression_results(models, val_reg_data, test_reg_data, probe_name, output_dir, 'regression')

        if "classification" in args.probe_types:
            test_cls_data = trainer.prepare_classification_data('test')
            test_cls_results, test_cls_preds = trainer.evaluate_probes(cls_probes, test_cls_data, 'classification')
            trainer.save_results(test_cls_results, test_cls_preds, output_dir, 'classification', 'test')

        if "regression_diff" in args.probe_types:
            val_reg_diff_data = trainer.prepare_regression_diff_data('val')
            test_reg_diff_data = trainer.prepare_regression_diff_data('test')
            test_reg_diff_results, test_reg_diff_preds = trainer.evaluate_probes(reg_diff_probes, test_reg_diff_data, 'regression_diff')
            trainer.save_results(test_reg_diff_results, test_reg_diff_preds, output_dir, 'regression_diff', 'test')

            for probe_name, models in reg_diff_probes.items():
                trainer.plot_regression_results(models, val_reg_diff_data, test_reg_diff_data, probe_name, output_dir, 'regression_diff')

    if args.cross_notation_eval:
        logger.info("="*60)
        logger.info("CROSS-NOTATION EVALUATION (BOTH DIRECTIONS)")
        logger.info("="*60)
        for eval_notation in trainer.notations:  # Evaluate on BOTH notations
            # Determine training notation (the other one)
            train_notation = [n for n in trainer.notations if n != eval_notation][0]
            
            logger.info(f"\n>>> Direction: {train_notation} → {eval_notation}")
            logger.info(f"    (Using probes trained on '{train_notation}', evaluating on '{eval_notation}' test data)")
            
            if "regression" in args.probe_types:      
                # Prepare val data for eval_notation only
                val_reg_data = trainer.prepare_regression_data('val')
                
                # Filter val data to only eval_notation samples
                filtered_val_data = {}
                for key in val_reg_data:
                    if key.startswith(f"{eval_notation}_"):
                        # Extract offset type: "sci_offset_0" -> "offset_0"
                        offset_type = key.split('_', 1)[1]
                        filtered_val_data[offset_type] = val_reg_data[key]

                # Prepare test data for eval_notation only
                test_reg_data = trainer.prepare_regression_data('test')
                
                # Filter test data to only eval_notation samples
                filtered_test_data = {}
                for key in test_reg_data:
                    if key.startswith(f"{eval_notation}_"):
                        # Extract offset type: "sci_offset_0" -> "offset_0"
                        offset_type = key.split('_', 1)[1]
                        filtered_test_data[offset_type] = test_reg_data[key]

                # Use probes trained on train_notation
                cross_probes = {}
                for key in reg_probes:
                    if key.startswith(f"{train_notation}_"):
                        offset_type = key.split('_', 1)[1]
                        cross_probes[offset_type] = reg_probes[key]
                
                test_results, test_preds = trainer.evaluate_probes(
                    cross_probes, filtered_test_data, 'regression')
                
                trainer.save_results(test_results, test_preds, output_dir, 
                                        'regression', 'test', subdir=f"cross_{train_notation}_to_{eval_notation}")
                
                # Generate scatter plots for cross-notation
                for offset_type, models in cross_probes.items():
                    if offset_type in filtered_test_data:
                        trainer.plot_regression_results(
                            models, filtered_val_data, filtered_test_data, offset_type, output_dir,
                            'regression', subdir=f"cross_{train_notation}_to_{eval_notation}")

    logger.info(f"All operations completed. Results in {output_dir}")


if __name__ == '__main__':
    main()