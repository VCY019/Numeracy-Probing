#!/usr/bin/env python3
"""
Embedding Extraction Script for Arxiv Numerical Data

This script extracts embeddings from language models for arxiv datasets with numerical values.
It extracts only offset_0 and offset_1 embeddings for numbers matching specified regex patterns.

Usage:
    python get_embeds_arxiv.py --data_path <path> --output_path <path> --model_path <path> --num_layers <int> --number_type <type>
"""

import os
import argparse
import json
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Number pattern definitions
DECIMAL_REGEX = r'(?<!\d\.)\d+\.\d+(?!\.\d)'
INT_REGEX = r'(?<![\d.])\d+(?![\d.])'
NUMERIC_BASE = r'(?:\d+(?:\.\d+)?)'
SCIENTIFIC_REGEX = NUMERIC_BASE + r'\s*×\s*10\s+[-+]?\d+'
NUMBER_PATTERNS = {
    'decimal': DECIMAL_REGEX,
    'scientific': SCIENTIFIC_REGEX,
    'integer': INT_REGEX,
}

def parse_numeric_value(value_str):
    """
    Parse numeric value from string, handling scientific notation.
    Returns the numeric value or None if parsing fails.
    """
    try:
        # Handle scientific notation with × symbol
        if "×" in value_str:
            parts = value_str.split("×")
            m = float(parts[0])
            n = float(parts[1].split()[-1])
            return m * (10 ** n)
        else:
            return float(value_str)
    except Exception as e:
        logger.warning(f"Failed to parse numeric value: '{value_str}' ({e})")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from arxiv numerical data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL dataset file")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument("--model_path", type=str, required=True, help="Path or name of the language model")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of model layers")
    parser.add_argument("--number_type", type=str, choices=['decimal', 'scientific', 'integer'], 
                       default='decimal', help="Type of numbers to extract")
    parser.add_argument("--finetuned_model", action="store_true", help="Whether the model is finetuned")
    parser.add_argument("--max_length", type=int, default=30000, help="Maximum document length to process")
    parser.add_argument("--max_numbers", type=int, default=5000, help="Maximum numbers to extract")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    if args.finetuned_model:
        logger.info(f"Loading finetuned model from {args.model_path}")
        peft_config = PeftConfig.from_pretrained(args.model_path)
        base_model_name = peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        logger.info(f"Loading base model from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)
    
    model.eval()
    
    # Create output directories
    number_dir = os.path.join(args.output_path, args.number_type)
    os.makedirs(os.path.join(number_dir, 'offset_0'), exist_ok=True)
    os.makedirs(os.path.join(number_dir, 'offset_1'), exist_ok=True)
    
    # Open output files
    offset_0_files = [open(os.path.join(number_dir, 'offset_0', f'layer_{i+1}.embeds'), 'wb') 
                      for i in range(args.num_layers)]
    offset_1_files = [open(os.path.join(number_dir, 'offset_1', f'layer_{i+1}.embeds'), 'wb') 
                      for i in range(args.num_layers)]

    metadata_file = open(os.path.join(number_dir, 'metadata.jsonl'), 'w')
    
    pattern = NUMBER_PATTERNS[args.number_type]
    total_numbers = 0
    
    doc_bar = tqdm(desc="Documents", unit="doc", position=0, leave=True)
    num_bar = tqdm(total=args.max_numbers, desc="Numbers extracted", unit="num", position=1, leave=True)

    try:
        with open(args.data_path, 'r') as f:
            for doc_idx, line in enumerate(f):
                if total_numbers >= args.max_numbers:
                    break
                doc = json.loads(line)
                text = doc['text']
                doc_bar.update(1)
                # Find all numbers in text
                matches = list(re.finditer(pattern, text))
                if not matches:
                    continue
                
                # Tokenize document
                encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
                input_ids = encoded['input_ids']
                offset_mapping = encoded['offset_mapping']
                
                if len(input_ids) > args.max_length:
                    logger.warning(f"Document {doc_idx} length {len(input_ids)} exceeds max length, skipping")
                    continue
                
                # Process through model
                input_tensor = torch.tensor([input_ids]).to(model.device)
                with torch.no_grad():
                    try:
                        outputs = model(input_ids=input_tensor, output_hidden_states=True, return_dict=True)
                    except Exception as e:
                        logger.error(f"Error processing document {doc_idx}: {e}")
                        continue
                
                # Extract embeddings for each number
                for match in matches:
                    num_bar.update(1)
                    if total_numbers >= args.max_numbers:
                        break
                    try:
                        # Find token indices
                        char_start, char_end = match.span()
                        number_value = match.group()
                        
                        # Parse numeric value
                        numeric_value = parse_numeric_value(number_value)
                        if numeric_value is None or numeric_value <= 0:
                            logger.warning(f"Skipping invalid numeric value: '{number_value}'")
                            continue
                        
                        # Find last token of number (offset_0)
                        last_char_pos = char_end - 1
                        try:
                            token_idx_0 = next(i for i, (start, end) in enumerate(offset_mapping) 
                                             if start <= last_char_pos < end)
                        except StopIteration:
                            logger.warning(f"Could not find token for number {number_value} at position {last_char_pos}")
                            continue
                        
                        # Find token after number (offset_1)
                        next_char_pos = char_end
                        try:
                            token_idx_1 = next(i for i, (start, end) in enumerate(offset_mapping) 
                                             if start <= next_char_pos < end)
                        except StopIteration:
                            token_idx_1 = len(offset_mapping) - 1
                        
                        # Extract and save embeddings for each layer
                        for layer_idx, layer_tensor in enumerate(outputs.hidden_states[1:]):
                            # offset_0 embedding
                            vector_0 = layer_tensor[0, token_idx_0, :].detach().cpu().numpy()
                            np.save(offset_0_files[layer_idx], vector_0)
                            
                            # offset_1 embedding  
                            vector_1 = layer_tensor[0, token_idx_1, :].detach().cpu().numpy()
                            np.save(offset_1_files[layer_idx], vector_1)
                        
                        # Save metadata with both string and numeric values
                        metadata = {
                            'global_id': total_numbers,
                            'doc_idx': doc_idx, 
                            'value': number_value,
                            'numeric_value': numeric_value,
                            'char_start': char_start,
                            'char_end': char_end
                        }
                        metadata_file.write(json.dumps(metadata) + '\n')
                        total_numbers += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing number {number_value} in doc {doc_idx}: {e}")
                        continue
                
                torch.cuda.empty_cache()
    
    finally:
        # Close all files
        for f in offset_0_files + offset_1_files:
            f.close()
        metadata_file.close()
    
    logger.info(f"Completed processing. Total numbers extracted: {total_numbers}")

if __name__ == "__main__":
    main() 