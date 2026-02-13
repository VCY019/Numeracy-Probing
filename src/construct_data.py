#!/usr/bin/env python3
"""
Data Construction Script for Numerical Comparison Datasets

Generates int-sci and dec-sci datasets with universal splits:
- 8000 train, 1600 val, 1600 test (stratified by digit length)

Usage:
    python construct_data.py --data_type int-sci --output_dir data/
    python construct_data.py --data_type dec-sci --output_dir data/
    python construct_data.py --all --output_dir data/
"""

import argparse
import json
import os
import random
from decimal import Decimal, getcontext
import logging
import math
from collections import OrderedDict

# Set decimal precision
getcontext().prec = 28

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def to_scientific(num):
    """Convert any positive number to scientific notation string."""
    if num == 0:
        return "0"
    
    exponent = math.floor(math.log10(abs(num)))
    base = num / (10 ** exponent)
    return f"{base:.5g} × 10^{exponent}" # 5 significant digits

def generate_int_sci_data(seed: int = 0) -> list:
    """
    Generate integer vs scientific notation comparison data.
    Returns 11200 examples (1400 per digit) with stratified splits
    """
    random.seed(seed)
    
    train_data = []
    val_data = []
    test_data = []
    
    # Generate 1400 examples per digit (2-9)
    for digit in range(2, 10):
        digit_data = []
        while len(digit_data) < 1400:
            rg = range(10**(digit-1), 10**digit)
            a = random.choice(rg)
            b = random.choice(rg)
            if a == b:
                continue
            
            # Convert randomly one to scientific notation
            if random.random() < 0.5:
                a_str = to_scientific(a)
                b_str = str(b)
                notations = ["sci", "int"]
            else:
                a_str = str(a)
                b_str = to_scientific(b)
                notations = ["int", "sci"]
            
            sample = {
                "digit": digit,
                "a": a_str,
                "b": b_str,
                "text": f"Which is larger, {a_str} or {b_str}?",
                "notations": notations
            }
            digit_data.append(sample)
        
        # Shuffle within digit
        random.shuffle(digit_data)
        
        # Stratified split: 1000 train, 200 val, 200 test per digit
        train_data.extend(digit_data[:1000])
        val_data.extend(digit_data[1000:1200])
        test_data.extend(digit_data[1200:])
    
    logger.info(f"Generated int-sci data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


def generate_dec_sci_data(seed: int = 0) -> list:
    """
    Generate decimal vs scientific notation comparison data.
    Returns 11200 examples (1400 per digit) with stratified splits.
    """
    random.seed(seed)
    
    train_data = []
    val_data = []
    test_data = []
    
    # Generate 1400 examples per digit (2-9)
    for digit in range(2, 10):
        digit_data = []
        while len(digit_data) < 1400:
            # Generate two numbers with specified digit length
            int_a = random.randint(10**(digit-1), 10**digit - 1)
            int_b = random.randint(10**(digit-1), 10**digit - 1)
            
            # Add decimal parts
            dec_len_a = random.randint(0, 4)
            if dec_len_a > 0:
                rand_int_a = random.randint(0, 10**dec_len_a - 1)
                dec_a = Decimal(str(rand_int_a)) / Decimal(10**dec_len_a)
            else:
                dec_a = Decimal('0')
            
            dec_len_b = random.randint(0, 4)
            if dec_len_b > 0:
                rand_int_b = random.randint(0, 10**dec_len_b - 1)
                dec_b = Decimal(str(rand_int_b)) / Decimal(10**dec_len_b)
            else:
                dec_b = Decimal('0')
            
            a = int_a + dec_a
            b = int_b + dec_b

            a_val = float(a)
            b_val = float(b)
            
            # Convert one to scientific notation randomly
            if random.random() < 0.5:
                a_str = to_scientific(a_val)
                b_str = str(b_val)
                notations = ["sci", "dec"]
            else:
                a_str = str(a_val)
                b_str = to_scientific(b_val)
                notations = ["dec", "sci"]
            
            sample = {
                "digit": digit,
                "a": a_str,
                "b": b_str,
                "text": f"Which is larger, {a_str} or {b_str}?",
                "notations": notations
            }
            digit_data.append(sample)
        
        # Shuffle within digit
        random.shuffle(digit_data)
        
        # Stratified split: 1000 train, 200 val, 200 test per digit
        train_data.extend(digit_data[:1000])
        val_data.extend(digit_data[1000:1200])
        test_data.extend(digit_data[1200:])
    
    logger.info(f"Generated dec-sci data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


def save_data(data: list, filepath: str) -> None:
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(data)} samples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate numerical comparison datasets")
    parser.add_argument("--data_type", type=str, 
                       choices=["int-sci", "dec-sci"],
                       help="Type of data to generate")
    parser.add_argument("--all", action="store_true", 
                       help="Generate all types of data")
    parser.add_argument("--output_dir", type=str, default="data/", 
                       help="Output directory for data files")
    parser.add_argument("--seed", type=int, default=0, 
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if not args.data_type and not args.all:
        parser.error("Must specify either --data_type or --all")
    
    # Define data types to generate
    if args.all:
        data_types = ["int-sci", "dec-sci"]
    else:
        data_types = [args.data_type]
    
    # Generate data for each type
    for data_type in data_types:
        logger.info(f"Generating {data_type} data...")
        
        if data_type == "int-sci":
            train_data, val_data, test_data = generate_int_sci_data(seed=args.seed)
            filepath = "int_sci_compare"
            
        elif data_type == "dec-sci":
            train_data, val_data, test_data = generate_dec_sci_data(seed=args.seed)
            filepath = "dec_sci_compare"
        
        # Save data
        filepath = os.path.join(args.output_dir, filepath)
        save_data(train_data, os.path.join(filepath, "train.jsonl"))
        save_data(val_data, os.path.join(filepath, "val.jsonl"))
        save_data(test_data, os.path.join(filepath, "test.jsonl"))
    
    logger.info("Data generation completed!")


if __name__ == "__main__":
    main()