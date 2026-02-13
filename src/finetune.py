#!/usr/bin/env python3
"""
finetune.py

Fine-tune a causal LLM on a number-comparison task while keeping
trainable linear probes (regression + classification + log-ratio regression).
"""

import argparse
import json
import re
import random
import os
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from sklearn.linear_model import Ridge, LogisticRegression
from tqdm import tqdm
import logging

from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumberComparisonDataset(Dataset):
    number_pattern = re.compile(
        r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[×x*]\s*10\^?-?\d+)?'
    )

    def __init__(self, data_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(data_path, 'r', encoding='utf8') as f:
            for line in f:
                record = json.loads(line)
                value_a = self._to_float(record['a'])
                value_b = self._to_float(record['b'])
                if value_a is None or value_b is None:
                    continue
                answer = record['a'] if value_a > value_b else record['b']
                prompt = f"Q: Which is larger, {record['a']} or {record['b']}? A: {answer}"
                encoding = tokenizer(
                    prompt,
                    padding='max_length',
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                idx_a, idx_b, idx_cls = self._find_token_indices(prompt)
                self.examples.append({
                    'input_ids': encoding['input_ids'][0],
                    'attention_mask': encoding['attention_mask'][0],
                    'value_a': value_a,
                    'value_b': value_b,
                    'log_value_a': float(np.log2(value_a)),
                    'log_value_b': float(np.log2(value_b)),
                    'log_ratio': float(np.log2(value_a) - np.log2(value_b)),
                    'comparison_label': 1.0 if value_a > value_b else 0.0,
                    'token_index_a': idx_a,
                    'token_index_b': idx_b,
                    'token_index_cls': idx_cls,
                })

    def _to_float(self, s: str):
        try:
            return float(eval(s.replace('×', '*').replace('^', '**').replace(',', '')))
        except Exception:
            return None

    def _find_token_indices(self, text: str):
        matches = list(re.finditer(self.number_pattern, text))
        if len(matches) < 3:
            raise ValueError(f"Expected 3 matches in {text}, got {len(matches)}")
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        offsets = encoding.offset_mapping
        indices = []
        for m in matches:
            char_end = m.end() - 1
            idx = next((i for i, (start, end) in enumerate(offsets) if start <= char_end < end), 0)
            indices.append(idx)

        # find the token index of the second colon in "A:"
        colon_indices = [i for i, token in enumerate(encoding.tokens()) if token == ':']
        if len(colon_indices) < 2:
            raise ValueError(f"Expected 2 colons in {text}, got {len(colon_indices)}")
        return indices[0], indices[1], colon_indices[1]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'input_ids': ex['input_ids'].clone(),
            'attention_mask': ex['attention_mask'].clone(),
            'value_a': torch.tensor(ex['value_a'], dtype=torch.float32),
            'value_b': torch.tensor(ex['value_b'], dtype=torch.float32),
            'log_value_a': torch.tensor(ex['log_value_a'], dtype=torch.float32),
            'log_value_b': torch.tensor(ex['log_value_b'], dtype=torch.float32),
            'log_ratio': torch.tensor(ex['log_ratio'], dtype=torch.float32),
            'comparison_label': torch.tensor(ex['comparison_label'], dtype=torch.float32),
            'token_index_a': ex['token_index_a'],
            'token_index_b': ex['token_index_b'],
            'token_index_cls': ex['token_index_cls'], # log ratio regressor use the same token index as the classification probe
        }


class LinearProbe(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def fit_sklearn_probes(model, loader, layers, device):
    """Fit sklearn probes for specified layers to initialize torch probes."""
    model.eval()
    reg_features = {l: [] for l in layers}
    cls_features = {l: [] for l in layers}
    reg_diff_features = {l: [] for l in layers}
    reg_targets = []
    cls_targets = []
    reg_diff_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Getting features for sklearn probes'):
            batch_device = {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}
            outputs = model(
                input_ids=batch_device['input_ids'],
                attention_mask=batch_device['attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
            hidden = outputs.hidden_states[1:]  # Skip embeddings

            bs = batch_device['input_ids'].size(0)
            batch_reg_targets = []
            batch_cls_targets = []
            batch_reg_diff_targets = []
            
            for i in range(bs):
                batch_reg_targets.extend([
                    batch['log_value_a'][i].item(),
                    batch['log_value_b'][i].item()
                ])
                batch_cls_targets.append(batch['comparison_label'][i].item())
                batch_reg_diff_targets.append(batch['log_ratio'][i].item())

            for layer in layers:
                h = hidden[layer].detach().cpu().numpy()
                for i in range(bs):
                    a_idx = batch['token_index_a'][i].item()
                    b_idx = batch['token_index_b'][i].item()
                    cls_idx = batch['token_index_cls'][i].item()
                    reg_features[layer].append(h[i, a_idx, :])
                    reg_features[layer].append(h[i, b_idx, :])
                    cls_features[layer].append(h[i, cls_idx, :])
                    reg_diff_features[layer].append(h[i, cls_idx, :])
            
            reg_targets.extend(batch_reg_targets)
            cls_targets.extend(batch_cls_targets)
            reg_diff_targets.extend(batch_reg_diff_targets)

    sklearn_probes = {}
    reg_targets = np.array(reg_targets)
    cls_targets = np.array(cls_targets)
    reg_diff_targets = np.array(reg_diff_targets)

    for layer in layers:
        X_reg = np.vstack(reg_features[layer])
        X_cls = np.vstack(cls_features[layer])
        X_reg_diff = np.vstack(reg_diff_features[layer])
        reg_model = Ridge().fit(X_reg, reg_targets)
        cls_model = LogisticRegression(max_iter=100).fit(X_cls, cls_targets)
        reg_diff_model = Ridge().fit(X_reg_diff, reg_diff_targets)
        
        sklearn_probes[layer] = {
            'regressor': reg_model, 
            'classifier': cls_model,
            'regressor_diff': reg_diff_model
        }
    
    return sklearn_probes


def custom_loss_function(lm_loss, reg_loss, cls_loss, reg_diff_loss, args):
    """
    Custom loss function using weighted sum of LM loss, regression MSE, and classification cross-entropy.
    """
    total_loss = (args.lm_loss_weight * lm_loss + 
                  args.regression_loss_weight * reg_loss + 
                  args.classification_loss_weight * cls_loss +
                  args.regression_diff_loss_weight * reg_diff_loss)
    
    return total_loss


def compute_probe_losses(hidden_states, batch, probes, layers, device):
    """
    Compute regression MSE and classification cross-entropy losses for each specified layer.
    Returns lists of losses per layer.
    """
    reg_loss_list = []
    cls_loss_list = []
    reg_diff_loss_list = []
    for layer in layers:
        h = hidden_states[layer]
        bs, seq_len, hidden_size = h.size()

        a_idx = batch['token_index_a']
        b_idx = batch['token_index_b']
        cls_idx = batch['token_index_cls']
        emb_a = h[torch.arange(bs), a_idx, :]
        emb_b = h[torch.arange(bs), b_idx, :]
        reg_feats = torch.cat([emb_a, emb_b], dim=0)
        reg_targets = torch.cat([batch['log_value_a'], batch['log_value_b']], dim=0).to(device)
        reg_feats = reg_feats.to(torch.float32)

        reg_preds = probes[str(layer)]['regression'](reg_feats)
        reg_loss_list.append(F.mse_loss(reg_preds, reg_targets))

        cls_feats = h[torch.arange(bs), cls_idx, :]
        cls_feats = cls_feats.to(torch.float32)
        cls_logits = probes[str(layer)]['classification'](cls_feats)
        cls_loss_list.append(F.binary_cross_entropy_with_logits(
            cls_logits, batch['comparison_label'].to(device)))
        
        reg_diff_feats = h[torch.arange(bs), cls_idx, :]
        reg_diff_feats = reg_diff_feats.to(torch.float32)
        reg_diff_preds = probes[str(layer)]['regression_diff'](reg_diff_feats)
        reg_diff_loss_list.append(F.mse_loss(reg_diff_preds, batch['log_ratio'].to(device)))

    return reg_loss_list, cls_loss_list, reg_diff_loss_list


def evaluate_model(model, loader, probe_dict, layers, device, args=None):
    """
    Evaluate the model on the validation set using simple metrics: average LM loss,
    torch regression MSE, and torch classification cross-entropy (using first probe layer).
    Returns a dict of results.
    """
    model.eval()
    sum_lm, sum_reg, sum_cls, sum_reg_diff = 0.0, 0.0, 0.0, 0.0 
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
            labels = batch_gpu['input_ids'].clone()
            labels[batch_gpu['attention_mask'] == 0] = -100
            labels = labels.to(device)
            out = model(
                input_ids=batch_gpu['input_ids'],
                attention_mask=batch_gpu['attention_mask'],
                labels=labels,
                output_hidden_states=True,
                return_dict=True
            )
            lm_loss = out.loss
            reg_loss_list, cls_loss_list, reg_diff_loss_list = compute_probe_losses(
                out.hidden_states[1:], batch_gpu, probe_dict, layers, device)
            bs = batch_gpu['input_ids'].size(0)

            sum_lm += lm_loss.item() * bs
            sum_reg += reg_loss_list[0].item() * bs
            sum_cls += cls_loss_list[0].item() * bs
            sum_reg_diff += reg_diff_loss_list[0].item() * bs
            count += bs

    results = {
        'language_model': sum_lm / count,
        'torch_regression_mse': sum_reg / count,
        'torch_classification_xent': sum_cls / count,
        'torch_regression_diff_mse': sum_reg_diff / count,
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    # === Data and Paths ===
    parser.add_argument('--train_data', required=True, help='Path to train.jsonl')
    parser.add_argument('--val_data', required=True, help='Path to val.jsonl')
    parser.add_argument('--save_path', default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--logs_path', default='./logs', help='Path to save scores as json')

    # === Model and Tokenizer ===
    parser.add_argument('--model_name', default='mistralai/Mistral-7B-v0.1', help='Model name or path')

    # === Training Hyperparameters ===
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for optimizer')
    parser.add_argument('--evaluate_every', type=int, default=200, help='Steps between evaluations')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_sequence_length', type=int, default=128, help='Maximum sequence length for tokenization')

    # === Probe and Loss Configuration ===
    parser.add_argument('--probe_layers', type=int, nargs='+', default=[0, -1], help='List of layer indices for probes')
    parser.add_argument('--lm_loss_weight', type=float, default=1.0, help='Weight for language modeling loss')
    parser.add_argument('--regression_loss_weight', type=float, default=0.1, help='Weight for regression probe loss')
    parser.add_argument('--classification_loss_weight', type=float, default=0.1, help='Weight for classification probe loss')
    parser.add_argument('--regression_diff_loss_weight', type=float, default=0, help='Weight for regression diff probe loss')
    parser.add_argument('--use_sklearn_init', action='store_true', help='Initialize torch probes from sklearn')

    # === LoRA Configuration ===
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for fine-tuning')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    args = parser.parse_args()
    
    # Print configuration with better formatting
    logger.info("=" * 80)
    logger.info("PROBE-AWARE FINETUNING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Validation data: {args.val_data}")
    logger.info(f"Probe layers: {args.probe_layers}")
    logger.info(f"Epochs: {args.num_epochs}, Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Loss weights - LM: {args.lm_loss_weight}, Reg: {args.regression_loss_weight}, \
          Cls: {args.classification_loss_weight}, Reg Diff: {args.regression_diff_loss_weight}")
    logger.info(f"LoRA: {args.use_lora}")
    if args.use_lora:
        logger.info(f"  LoRA config - r: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    logger.info(f"Sklearn init: {args.use_sklearn_init}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info("=" * 80)
    
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.logs_path, exist_ok=True)
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    set_seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = NumberComparisonDataset(args.train_data, tokenizer, args.max_sequence_length)
    val_ds = NumberComparisonDataset(args.val_data, tokenizer, args.max_sequence_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f'Train examples: {len(train_ds)}, Val examples: {len(val_ds)}')

    model_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=model_dtype)
    
    if args.use_lora:
        logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Initial sklearn probe fitting for initialization
    initial_sklearn_probes = None
    if args.regression_loss_weight == 0 and args.classification_loss_weight == 0 and args.regression_diff_loss_weight == 0:
        args.use_sklearn_init = False
    if args.use_sklearn_init:
        logger.info("=" * 50)
        logger.info("FITTING INITIAL SKLEARN PROBES")
        logger.info("=" * 50)
        initial_sklearn_probes = fit_sklearn_probes(model, train_loader, args.probe_layers, device)
        logger.info("=" * 50)
        logger.info("INITIAL SKLEARN PROBES FITTED")
        logger.info("=" * 50)

    hidden_size = model.config.hidden_size
    torch_probes = {}
    for layer in args.probe_layers:
        torch_probes[str(layer)] = {
            'regression': LinearProbe(hidden_size).to(device).to(torch.float32),
            'regression_diff': LinearProbe(hidden_size).to(device).to(torch.float32),
            'classification': LinearProbe(hidden_size).to(device).to(torch.float32)
        }

    if args.use_sklearn_init and initial_sklearn_probes is not None:
        logger.info("Initializing torch probes from sklearn...")
        for layer in args.probe_layers:
            coef_r = initial_sklearn_probes[layer]['regressor'].coef_
            int_r = initial_sklearn_probes[layer]['regressor'].intercept_
            torch_probes[str(layer)]['regression'].linear.weight.data.copy_(
                torch.from_numpy(coef_r).view(1, -1))
            torch_probes[str(layer)]['regression'].linear.bias.data.copy_(
                torch.from_numpy(np.array([int_r])))
            
            coef_c = initial_sklearn_probes[layer]['classifier'].coef_
            int_c = initial_sklearn_probes[layer]['classifier'].intercept_
            torch_probes[str(layer)]['classification'].linear.weight.data.copy_(
                torch.from_numpy(coef_c).view(1, -1))
            torch_probes[str(layer)]['classification'].linear.bias.data.copy_(
                torch.from_numpy(int_c))
            
            coef_r_diff = initial_sklearn_probes[layer]['regressor_diff'].coef_
            int_r_diff = initial_sklearn_probes[layer]['regressor_diff'].intercept_
            torch_probes[str(layer)]['regression_diff'].linear.weight.data.copy_(
                torch.from_numpy(coef_r_diff).view(1, -1))
            torch_probes[str(layer)]['regression_diff'].linear.bias.data.copy_(
                torch.from_numpy(np.array([int_r_diff])))
            
        logger.info('✓ Torch probes initialized from sklearn.')
    else:
        logger.info('✓ Torch probes initialized randomly.')
    
    # Evaluate the model before finetuning
    logger.info("\n" + "=" * 50)
    logger.info("INITIAL EVALUATION (BEFORE FINETUNING)")
    logger.info("=" * 50)
    gc.collect()
    initial_eval_results = evaluate_model(
        model, val_loader, torch_probes,
        args.probe_layers, device, args=args
    )
    gc.collect()
    
    initial_total_loss = custom_loss_function(
        initial_eval_results['language_model'], 
        initial_eval_results['torch_regression_mse'], 
        initial_eval_results['torch_classification_xent'], 
        initial_eval_results['torch_regression_diff_mse'],
        args
    )
    
    logger.info(f"Initial evaluation results:")
    logger.info(f"  Language Model Loss: {initial_eval_results['language_model']:.4f}")
    logger.info(f"  Regression MSE: {initial_eval_results['torch_regression_mse']:.4f}")
    logger.info(f"  Classification XEnt: {initial_eval_results['torch_classification_xent']:.4f}")
    logger.info(f"  Regression Diff MSE: {initial_eval_results['torch_regression_diff_mse']:.4f}")
    logger.info(f"  Total Loss: {initial_total_loss:.4f}")
    logger.info("=" * 50)

    all_params = list(model.parameters())
    for layer in args.probe_layers:
        all_params += list(torch_probes[str(layer)]['regression'].parameters())
        all_params += list(torch_probes[str(layer)]['classification'].parameters())
        all_params += list(torch_probes[str(layer)]['regression_diff'].parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.learning_rate)

    best_total_loss = float('inf')
    step = 0
    metrics_log = {}
    
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    for epoch in range(args.num_epochs):
        logger.info(f'\n{"="*20} Epoch {epoch+1}/{args.num_epochs} {"="*20}')
        model.train()
        
        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'): 
            step += 1
            batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
            labels = batch_gpu['input_ids'].clone()
            labels[batch_gpu['attention_mask'] == 0] = -100
            labels = labels.to(device)
            out = model(
                input_ids=batch_gpu['input_ids'],
                attention_mask=batch_gpu['attention_mask'],
                labels=labels,
                output_hidden_states=True,
                return_dict=True
            )
            lm_loss = out.loss
            reg_loss_list, cls_loss_list, reg_diff_loss_list = compute_probe_losses(
                out.hidden_states[1:], batch_gpu, torch_probes, args.probe_layers, device)

            total_loss = custom_loss_function(lm_loss, reg_loss_list[0], cls_loss_list[0], reg_diff_loss_list[0], args)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % args.evaluate_every == 0:
                logger.info(f'\n{"-"*20} Evaluation at Step {step} {"-"*20}')
                gc.collect()
                eval_results = evaluate_model(
                    model, val_loader, torch_probes,
                    args.probe_layers, device, args=args
                )
                gc.collect()
                eval_total_loss = custom_loss_function(
                    eval_results['language_model'], 
                    eval_results['torch_regression_mse'], 
                    eval_results['torch_classification_xent'], 
                    eval_results['torch_regression_diff_mse'],
                    args
                )
                
                logger.info(f"Step {step} Results:")
                logger.info(f"  Train Total Loss: {total_loss.item():.4f}")
                logger.info(f"  Val Total Loss:   {eval_total_loss:.4f}")
                logger.info(f"  Val LM Loss:      {eval_results['language_model']:.4f}")
                logger.info(f"  Val Reg MSE:      {eval_results['torch_regression_mse']:.4f}")
                logger.info(f"  Val Cls XEnt:     {eval_results['torch_classification_xent']:.4f}")
                logger.info(f"  Val Reg Diff MSE: {eval_results['torch_regression_diff_mse']:.4f}")

                metrics_log[step] = {
                    "total_loss": float(eval_total_loss),
                    **{k: float(v) for k, v in eval_results.items()}
                }
                
                if eval_total_loss < best_total_loss:
                    best_total_loss = eval_total_loss
                    save_path = os.path.join(args.save_path, 'best_model')
                    model.save_pretrained(save_path)
                    logger.info(f"  → New best model saved: {save_path}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Best validation total loss: {best_total_loss:.4f}")
    
    with open(os.path.join(args.logs_path, 'metrics_log.json'), 'w') as f:
        json.dump(metrics_log, f, indent=4)
    logger.info(f"Metrics saved to: {os.path.join(args.logs_path, 'metrics_log.json')}")
    logger.info("=" * 80)
    
    
if __name__=='__main__':
    main()