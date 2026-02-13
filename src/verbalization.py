#!/usr/bin/env python3
"""
Verbalization Testing Script for Numerical Comparison Datasets

This script tests language models' ability to compare numbers in different formats
by directly querying the model and evaluating its responses.

Supported data types (all are LEGACY except for int-sci and dec-sci):
- int-dec: Integer vs Decimal comparison
- int-sci: Integer vs Scientific notation comparison
- dec-dec: Decimal vs Decimal comparison
- dec-dec-hard: Hard decimal comparison (same integer part)
- dec-sci: Decimal vs Scientific notation comparison
- sci-sci: Scientific notation vs Scientific notation comparison

Features (supported for int-sci and dec-sci datasets)
- In-context learning (ICL) with 1-5 shot examples 
- Supports both "larger" and "smaller" comparison operators
- Alternative prompt ordering with --use_alt_prompt (swap order of numbers in ICL examples)

Usage:
    python verbalization.py --data_path <path> --model_path <path> --output_path <path>

Example:
    python verbalization.py --data_path data/int_sci_compare/test.jsonl --model_path meta-llama/Llama-2-7b-hf --output_path verbalization/Llama-2-7b-hf/
"""

import argparse
import json
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from peft import PeftModel, PeftConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VerbalizationTester:
    """Class for testing numerical comparison through verbalization."""

    def __init__(
        self, model_path: str, use_icl: bool = True, finetuned_model: bool = False,
        use_alt_prompt: bool = False, n_few_shot: int = 1, operator: str = "larger"
    ):
        """Initialize the verbalization tester with model and tokenizer."""
        self.model_path = model_path
        self.use_icl = use_icl
        self.use_alt_prompt = use_alt_prompt
        self.n_few_shot = n_few_shot
        assert operator in ["smaller", "larger"], f"Unsupported operator {operator}"
        self.operator = operator

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        if finetuned_model:
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name = peft_config.base_model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
        self.model.eval()

        # Regex pattern to extract numbers from model outputs
        self.number_regex = r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[×x*]\s*10\^?-?\d+)?'

    def query_model(self, prompt: str) -> str:
        """Query the model with a prompt and return the response."""
        input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**input_ids, max_new_tokens=40, do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def convert_val(self, val_str: str) -> float:
        """Convert a value string (including scientific notation) to float."""
        try:
            return float(eval(val_str.replace('×', '*').replace('^', '**').replace(',', '')))
        except:
            return None

    def extract_answer(self, output: str) -> str:
        """Extract the numerical answer from model output."""
        try:
            # Get the answer immediately after the prompt
            answer_part = output.split('A:')[self.n_few_shot+1].strip()
            # Find the first number in the answer
            numbers = re.findall(self.number_regex, answer_part)
            if numbers:
                return numbers[0]
            return None
        except:
            return None

    def create_comparison_prompt(self, a: str, b: str, data_type: str) -> str:
        """Create a comparison prompt with appropriate in-context learning examples."""
        base_prompt = f"Q: Which is larger, {a} or {b}? A:"

        if not self.use_icl:
            return base_prompt

        ## LEGACY prompt creation. Overridden for int-sci and dec-sci

        # Add appropriate ICL examples based on data type
        icl_template = "Q: Which is larger, {first_num} or {second_num}? A: {answer_num}"
        if data_type == "int-sci":
            first_num, second_num = "9.9 × 10^2", "100"
            answer_num = first_num
        elif data_type == "dec-sci":
            first_num, second_num = "899.9", "9.9 × 10^2"
            answer_num = second_num
        elif data_type == "sci-sci":
            first_num, second_num = "1.0 × 10^2", "9.9 × 10^1"
            answer_num = first_num
        elif data_type == "dec-dec-hard":
            first_num, second_num = "9.9", "9.11"
            answer_num = first_num
        elif data_type == "int-dec":
            first_num, second_num = "650", "649.73"
            answer_num = first_num
        elif data_type == "dec-dec":
            first_num, second_num = "343.2", "245.195"
            answer_num = first_num

        if not self.use_alt_prompt:
            icl_example = icl_template.format(
                first_num=first_num, second_num=second_num, answer_num=answer_num)
        else:
            icl_example = icl_template.format(
                first_num=second_num, second_num=first_num, answer_num=answer_num)

        # NEW few-shot example prompt.
        if data_type in ["int-sci", "dec-sci"]:
            base_prompt = f"Q: Which is {self.operator}, {a} or {b}? A:"
            icl_template = "Q: Which is {operator}, {first_num} or {second_num}? A: {answer_num}"
            assert 1 <= self.n_few_shot <= 5

            if data_type == "int-sci":
                pairs = [
                    ("9.9 × 10^2", "100"),  # 0
                    ("161230", "7.182 × 10^5"),  # 1
                    ("713", "4.78 × 10^2"),  # 0
                    ("1.354 × 10^6", "4906723"),  # 1
                    ("20834", "6.5 × 10^3"),  # 0
                ][:self.n_few_shot]
                answer_ids = [0, 1, 0, 1, 0][:self.n_few_shot]
            elif data_type == "dec-sci":
                pairs = [
                    ("9.9 × 10^2", "899.9"),  # 0
                    ("161230.51", "7.182 × 10^5"),  # 1
                    ("712.34", "4.78 × 10^2"),  # 0
                    ("1.354 × 10^6", "4906723.2"),  # 1
                    ("20834.17033", "6.5 × 10^3"),  # 0
                ][:self.n_few_shot]
                answer_ids = [0, 1, 0, 1, 0][:self.n_few_shot]
            else:
                raise ValueError(data_type)

            icl_examples = []
            for (a_str, b_str), ans_id in zip(pairs, answer_ids):
                if self.operator == "smaller":
                    ans_id = 1 - ans_id
                ans = [a_str, b_str][ans_id]

                if not self.use_alt_prompt:
                    icl_examples.append(icl_template.format(operator=self.operator, first_num=a_str, second_num=b_str, answer_num=ans))
                else:
                    icl_examples.append(icl_template.format(operator=self.operator, first_num=b_str, second_num=a_str, answer_num=ans))

            icl_example = "\n".join(icl_examples)
        else:
            assert self.n_few_shot == 1

        return f"{icl_example}\n{base_prompt}"

    def test_sample(self, sample: dict, data_type: str) -> dict:
        """Test a single sample and return results."""
        a, b = sample["a"], sample["b"]
        true_a = self.convert_val(a)
        true_b = self.convert_val(b)

        # Create prompt and query model
        prompt = self.create_comparison_prompt(a, b, data_type)
        output = self.query_model(prompt)

        # Extract model's answer
        model_answer = self.extract_answer(output)
        model_value = self.convert_val(model_answer) if model_answer else None

        # Determine correctness
        comparison_correct = False
        if model_value is not None:
            if (true_a > true_b and abs(model_value - true_a) < 1e-3) or \
               (true_b > true_a and abs(model_value - true_b) < 1e-3):
                comparison_correct = True

            if self.operator == "smaller":
                comparison_correct = (
                    (true_a < true_b and abs(model_value - true_a) < 1e-3) or
                    (true_b < true_a and abs(model_value - true_b) < 1e-3)
                )

        return {
            "id": sample.get("id", 0),
            "digit": sample.get("digit", None),
            "a": a,
            "b": b,
            "is_prefix": sample.get("is_prefix", False),
            "model_output": output,
            "comparison_correct": comparison_correct,
        }

    def test_dataset(self, test_data_path: str, data_type: str) -> dict:
        """
        Test the dataset and return comprehensive results including accuracy metrics.

        Args:
            test_data_path: Path to the JSONL data file
            data_type: Type of comparison data (e.g., "int-sci", "dec-sci", etc.)

        Returns:
            Dictionary containing summary statistics, detailed results, and error logs
        """
        # Load data
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = [json.loads(line.strip()) for line in f]

        logger.info(f"Testing {len(test_data)} samples")

        # Test all samples
        results = []
        error_logs = []

        for sample in tqdm(test_data, desc="Testing samples"):
            try:
                result = self.test_sample(sample, data_type)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                error_logs.append({
                    "sample_id": sample.get("id", "unknown"),
                    "error": str(e),
                    "sample": sample
                })

        # Calculate accuracies
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r["comparison_correct"])
        overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

        # Calculate accuracy by digit length if applicable
        digit_accuracies = {}
        if any(r.get("digit") is not None for r in results):
            for result in results:
                digit = result.get("digit")
                if digit is not None:
                    if digit not in digit_accuracies:
                        digit_accuracies[digit] = {"correct": 0, "total": 0}
                    digit_accuracies[digit]["total"] += 1
                    if result["comparison_correct"]:
                        digit_accuracies[digit]["correct"] += 1

            # Convert to percentages
            for digit in digit_accuracies:
                total = digit_accuracies[digit]["total"]
                correct = digit_accuracies[digit]["correct"]
                digit_accuracies[digit]["accuracy"] = correct / total if total > 0 else 0.0

        return {
            "summary": {
                "total_samples": total_samples,
                "correct_samples": correct_samples,
                "overall_accuracy": overall_accuracy,
                "digit_accuracies": digit_accuracies,
                "error_count": len(error_logs)
            },
            "detailed_results": results,
            "error_logs": error_logs
        }


def get_data_type_from_filename(filename: str) -> str:
    """Infer data type from filename."""
    filename = filename.lower()
    if "int_dec" in filename or "int-dec" in filename:
        return "int-dec"
    elif "int_sci" in filename or "int-sci" in filename:
        return "int-sci"
    elif "dec_dec_hard" in filename or "dec-dec-hard" in filename:
        return "dec-dec-hard"
    elif "dec_dec" in filename or "dec-dec" in filename:
        return "dec-dec"
    elif "dec_sci" in filename or "dec-sci" in filename:
        return "dec-sci"
    elif "sci_sci" in filename or "sci-sci" in filename:
        return "sci-sci"
    else:
        raise ValueError(f"Unknown data type: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Test numerical comparison through verbalization")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the input JSONL dataset file")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path or name of the language model")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--data_type", type=str, default=None,
                       choices=["int-dec", "int-sci", "dec-dec", "dec-dec-hard", "dec-sci", "sci-sci"],
                       help="Type of numerical comparison data (auto-detected if not specified)")
    parser.add_argument("--use_icl", action="store_true", default=True,
                       help="Use in-context learning examples")
    parser.add_argument("--use_alt_prompt", action="store_true", default=False, help="Use alternative prompt (swap order of numbers in ICL examples)")
    parser.add_argument("--n_few_shot", type=int, default=1, help="Number of few shot examples.")
    parser.add_argument("--operator", type=str, default="larger", choices=["smaller", "larger"],
                        help="Choose whether to ask for the larger number or the smaller one.")
    parser.add_argument("--no_icl", action="store_true",
                       help="Disable in-context learning examples")
    parser.add_argument("--finetuned_model", action="store_true",
                        help="Whether the model is finetuned")
    args = parser.parse_args()

    # Handle ICL flag
    if args.no_icl:
        args.use_icl = False

    # Auto-detect data type if not specified
    if args.data_type is None:
        args.data_type = get_data_type_from_filename(args.data_path)
        logger.info(f"Auto-detected data type: {args.data_type}")

    # Validate input file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Use ICL: {args.use_icl}")
    logger.info(f"Use alternative prompt: {args.use_alt_prompt}")
    logger.info(f"Num. of few shot examples: {args.n_few_shot}")
    logger.info(f"Operator: {args.operator}")

    # Initialize tester and run evaluation
    tester = VerbalizationTester(
        model_path=args.model_path, use_icl=args.use_icl, finetuned_model=args.finetuned_model,
        use_alt_prompt=args.use_alt_prompt, n_few_shot=args.n_few_shot, operator=args.operator)
    results = tester.test_dataset(
        test_data_path=args.data_path,
        data_type=args.data_type,
    )

    # Save results
    data_name = args.data_type.replace("-", "_") + "_compare"
    output_file = os.path.join(args.output_path, f"{data_name}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        # Write summary
        f.write(json.dumps(results["summary"], ensure_ascii=False) + "\n")
        # Write error logs
        f.write(json.dumps(results["error_logs"], ensure_ascii=False) + "\n")
        # Write detailed results
        for result in results["detailed_results"]:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Print summary
    summary = results["summary"]
    logger.info("=" * 50)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total samples: {summary['total_samples']}")
    logger.info(f"Correct samples: {summary['correct_samples']}")
    logger.info(f"Overall accuracy: {summary['overall_accuracy']}")
    logger.info(f"Errors: {summary['error_count']}")

    if summary['digit_accuracies']:
        logger.info("\nAccuracy by digit length:")
        for digit in sorted(summary['digit_accuracies'].keys()):
            acc_info = summary['digit_accuracies'][digit]
            logger.info(f"  Digit {digit}: {acc_info['accuracy']} ({acc_info['correct']}/{acc_info['total']})")

    logger.info(f"\nDetailed results saved to: {output_file}")
    logger.info("Verbalization testing completed successfully!")


if __name__ == "__main__":
    main()
