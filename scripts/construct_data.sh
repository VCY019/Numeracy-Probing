#!/bin/bash

echo "Generating all numerical comparison datasets..."

# Navigate to repository root and run Python script
python src/construct_data.py --all --output_dir data/

echo "Dataset generation completed!"