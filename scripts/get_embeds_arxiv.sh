#!/bin/bash
#SBATCH --job-name=get_embeds_arxiv
#SBATCH --output=sbatch_logs/get_embeds_arxiv_%j.log
#SBATCH --error=sbatch_logs/get_embeds_arxiv_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1    

python src/get_embeds_arxiv.py \
    --data_path data/arxiv_100k.jsonl \
    --output_path embeddings_arxiv/Mistral-7B-v0.1 \
    --model_path mistralai/Mistral-7B-v0.1 \
    --num_layers 32 \
    --number_type decimal \
    --max_numbers 5000 \
    --max_length 30000

python src/get_embeds_arxiv.py \
    --data_path data/arxiv_100k.jsonl \
    --output_path embeddings_arxiv/Mistral-7B-v0.1 \
    --model_path mistralai/Mistral-7B-v0.1 \
    --num_layers 32 \
    --number_type scientific \
    --max_numbers 5000 \
    --max_length 30000