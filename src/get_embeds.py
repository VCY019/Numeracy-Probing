#!/usr/bin/env python3
"""
Embedding Extraction Script for Numerical Comparison Datasets

This script extracts embeddings from language models for numerical comparison datasets.
It can selectively extract three types of internal states:
1. offset_0: vectors of the last token of numerals
2. offset_1: vectors of the token after the numerals  
3. last_token: vector of the last token of the prompt

Usage:
    python get_embeds.py --data_path <path> --output_path <path> --model_path <path> --num_layers <int> [--embed_types <types>]

Examples:
    # Extract all embedding types (default)
    python get_embeds.py --data_path data/int_sci_compare/train.jsonl --output_path embeddings/Mistral-7B-v0.1/int_sci_compare/train --model_path meta-llama/Llama-2-7b-hf --num_layers 32
    
    # Extract only offset_0 and offset_1 embeddings
    python get_embeds.py --data_path data/int_sci_compare/train.jsonl --output_path embeddings/Mistral-7B-v0.1/int_sci_compare/train --model_path meta-llama/Llama-2-7b-hf --num_layers 32 --embed_types offset_0 offset_1
    
    # Extract only last_token embeddings
    python get_embeds.py --data_path data/int_sci_compare/train.jsonl --output_path embeddings/Mistral-7B-v0.1/int_sci_compare/train --model_path meta-llama/Llama-2-7b-hf --num_layers 32 --embed_types last_token
"""

import argparse
import json
import os
import re
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from peft import PeftModel, PeftConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Class for extracting embeddings from numerical comparison datasets."""
    
    def __init__(self, model_path: str, num_layers: int, finetuned_model: bool, embed_types: list = None):
        """Initialize the embedding extractor with model and tokenizer."""
        self.model_path = model_path
        self.num_layers = num_layers
        self.embed_types = embed_types or ["all"]
        
        # Determine which embedding types to extract
        if "all" in self.embed_types:
            self.extract_offset_0 = True
            self.extract_offset_1 = True 
            self.extract_last_token = True
        else:
            self.extract_offset_0 = "offset_0" in self.embed_types
            self.extract_offset_1 = "offset_1" in self.embed_types
            self.extract_last_token = "last_token" in self.embed_types
        
        logger.info(f"Extracting embedding types: offset_0={self.extract_offset_0}, offset_1={self.extract_offset_1}, last_token={self.extract_last_token}")

        if finetuned_model:
            logging.info(f"Loading finetuned model from {model_path}")
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name = peft_config.base_model_name_or_path
            logging.info(f"Base model name: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            logging.info(f"Loading base model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
        self.model.eval()
        
        # Comprehensive regex pattern to match all number formats
        self.NUM_PATTERN = r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[×x*]\s*10\^?-?\d+)?'
    
    def find_number_spans(self, text: str) -> list:
        """Find all number spans in the text."""
        matches = list(re.finditer(self.NUM_PATTERN, text))
        return [(m.start(), m.end(), m.group()) for m in matches]
    
    def get_token_indices(self, text: str, char_spans: list, offset_mapping: list) -> dict:
        """Get token indices for character spans."""
        token_indices = {
            'offset_0': [],  # last token of number
            'offset_1': []   # token after number
        }
        
        for char_start, char_end, number_text in char_spans:
            # Find last token of the number (offset_0)
            last_char_pos = char_end - 1
            try:
                token_idx = next(
                    i for i, (start, end) in enumerate(offset_mapping)
                    if start <= last_char_pos < end
                )
                token_indices['offset_0'].append(token_idx)
            except StopIteration:
                logger.warning(f"Could not find token for character position {last_char_pos} in '{number_text}'")
                continue
            
            # Find token after the number (offset_1)
            next_char_pos = char_end
            try:
                token_idx_offset_1 = next(
                    i for i, (start, end) in enumerate(offset_mapping)
                    if start <= next_char_pos < end
                )
                token_indices['offset_1'].append(token_idx_offset_1)
            except StopIteration:
                # If no token after, use the last token of the sequence
                token_indices['offset_1'].append(len(offset_mapping) - 1)
        
        return token_indices
    
    def extract_embeddings_from_sample(self, sample: dict) -> dict:
        """Extract embeddings from a single sample."""
        text = sample["text"]
        
        # Tokenize the text
        encoded = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        offset_mapping = encoded["offset_mapping"]
        
        # Only find number spans if we need offset embeddings
        if self.extract_offset_0 or self.extract_offset_1:
            number_spans = self.find_number_spans(text)
            
            if len(number_spans) < 2:
                logger.warning(f"Expected 2 numbers but found {len(number_spans)} in: {text}")
                return None
            
            # Get token indices for the numbers
            token_indices = self.get_token_indices(text, number_spans, offset_mapping)
        else:
            token_indices = {'offset_0': [], 'offset_1': []}
        
        # Process through model
        input_tensor = torch.tensor([input_ids]).to(self.model.device)
        
        with torch.no_grad():
            try:
                outputs = self.model(input_ids=input_tensor, output_hidden_states=True, return_dict=True)
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                return None
        
        # Extract embeddings for each layer
        embeddings = {}
        if self.extract_offset_0:
            embeddings['offset_0'] = []
        if self.extract_offset_1:
            embeddings['offset_1'] = []
        if self.extract_last_token:
            embeddings['last_token'] = []
        
        hidden_states = outputs.hidden_states[1:]  # Skip input embeddings
        last_token_idx = input_tensor.shape[1] - 1
        
        for layer_idx, layer_tensor in enumerate(hidden_states):
            layer_embeddings = {}
            
            if self.extract_offset_0:
                layer_embeddings['offset_0'] = []
                for token_idx in token_indices['offset_0']:
                    vector = layer_tensor[0, token_idx, :].detach().cpu().numpy()
                    layer_embeddings['offset_0'].append(vector)
                embeddings['offset_0'].append(layer_embeddings['offset_0'])
            
            if self.extract_offset_1:
                layer_embeddings['offset_1'] = []
                for token_idx in token_indices['offset_1']:
                    vector = layer_tensor[0, token_idx, :].detach().cpu().numpy()
                    layer_embeddings['offset_1'].append(vector)
                embeddings['offset_1'].append(layer_embeddings['offset_1'])
            
            if self.extract_last_token:
                layer_embeddings['last_token'] = layer_tensor[0, last_token_idx, :].detach().cpu().numpy()
                embeddings['last_token'].append(layer_embeddings['last_token'])
        
        return embeddings
    
    def process_dataset(self, data_path: str, output_path: str) -> None:
        """Process entire dataset and save embeddings."""
        # Create output directories only for specified embedding types
        active_embed_types = []
        if self.extract_offset_0:
            active_embed_types.append('offset_0')
        if self.extract_offset_1:
            active_embed_types.append('offset_1')
        if self.extract_last_token:
            active_embed_types.append('last_token')
        
        for embed_type in active_embed_types:
            os.makedirs(os.path.join(output_path, embed_type), exist_ok=True)
        
        # Count total samples
        total_samples = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for _ in f:
                total_samples += 1
        
        # Create output file handles only for active embedding types
        output_files = {}
        for embedding_type in active_embed_types:
            output_files[embedding_type] = [
                open(os.path.join(output_path, embedding_type, f"layer_{layer_idx+1}.embeds"), "wb")
                for layer_idx in range(self.num_layers)
            ]
        
        try:
            # Process each sample
            with open(data_path, "r", encoding="utf-8") as f:
                for sample_idx, line in enumerate(tqdm(f, desc="Processing samples", total=total_samples)):
                    sample = json.loads(line.strip())
                    
                    # Extract embeddings
                    embeddings = self.extract_embeddings_from_sample(sample)
                    if embeddings is None:
                        continue
                    
                    # Save embeddings for each layer
                    for layer_idx in range(self.num_layers):
                        # Save offset_0 embeddings (both numbers) if extracted
                        if self.extract_offset_0 and 'offset_0' in embeddings:
                            for num_embedding in embeddings['offset_0'][layer_idx]:
                                np.save(output_files['offset_0'][layer_idx], num_embedding)
                        
                        # Save offset_1 embeddings (both numbers) if extracted
                        if self.extract_offset_1 and 'offset_1' in embeddings:
                            for num_embedding in embeddings['offset_1'][layer_idx]:
                                np.save(output_files['offset_1'][layer_idx], num_embedding)
                        
                        # Save last token embedding if extracted
                        if self.extract_last_token and 'last_token' in embeddings:
                            np.save(output_files['last_token'][layer_idx], embeddings['last_token'][layer_idx])
        
        finally:
            # Close all file handles
            for embedding_type in output_files:
                for file_handle in output_files[embedding_type]:
                    file_handle.close()
        
        logger.info(f"Completed processing {total_samples} samples")
        logger.info(f"Embeddings saved to {output_path}")
        logger.info(f"Extracted embedding types: {active_embed_types}")


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from numerical comparison datasets")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the input JSONL dataset file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output directory for embeddings")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path or name of the language model")
    parser.add_argument("--num_layers", type=int, required=True,
                       help="Number of model layers")
    parser.add_argument("--finetuned_model", action="store_true",
                        help="Whether the model is finetuned")
    parser.add_argument(
        "--embed_types", type=str, nargs='+', default=["all"],
        help="Embedding types to extract. Options: 'all', 'offset_0', 'offset_1', 'last_token' (default: ['all'])"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Using {args.num_layers} layers for model {args.model_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Embedding types to extract: {args.embed_types}")
    
    # Validate input file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize extractor and process dataset
    extractor = EmbeddingExtractor(
        model_path=args.model_path, 
        num_layers=args.num_layers, 
        finetuned_model=args.finetuned_model,
        embed_types=args.embed_types
    )
    extractor.process_dataset(data_path=args.data_path, output_path=args.output_path)
    
    logger.info("Embedding extraction completed successfully!")


if __name__ == "__main__":
    main()
