#!/bin/bash
#SBATCH --job-name=train_probe_arxiv
#SBATCH --output=sbatch_logs/train_probe_arxiv_%j.log
#SBATCH --error=sbatch_logs/train_probe_arxiv_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=ica100  
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1    

python src/train_probe_arxiv.py \
    --data_path_decimal embeddings_arxiv/Mistral-7B-v0.1/decimal \
    --data_path_scientific embeddings_arxiv/Mistral-7B-v0.1/scientific \
    --model_name Mistral-7B-v0.1 \
    --num_layers 32