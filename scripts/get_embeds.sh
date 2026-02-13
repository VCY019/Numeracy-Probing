#!/bin/bash
#SBATCH --job-name=get_embeds
#SBATCH --output=sbatch_logs/get_embeds_%j.log
#SBATCH --error=sbatch_logs/get_embeds_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1

# Data name and split
DATA_NAME=(
    "int_sci_compare:train"
    "int_sci_compare:val"
    "int_sci_compare:test"
    "dec_sci_compare:train"
    "dec_sci_compare:val"
    "dec_sci_compare:test"
)


# Model configurations with layers (format: "model_path:num_layers")
MODEL_CONFIGS=(
    "mistralai/Mistral-7B-v0.1:32"
    "allenai/OLMo-2-1124-7B-Instruct:32"
    "meta-llama/Llama-2-7b-hf:32"
    "Qwen/Qwen3-8B:36"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:28"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B:32"
    "meta-llama/Llama-3.1-8B-Instruct:32"
)

# Loop through all combinations
for data_name in "${DATA_NAME[@]}"; do
    # Split DATA_NAME into dataset name and split
    IFS=':' read -r dataset_name split <<< "$data_name"
    
    for model_config in "${MODEL_CONFIGS[@]}"; do
        # Split MODEL_CONFIGS into model path and number of layers
        IFS=':' read -r model_path num_layers <<< "$model_config"

        # Extract model name for output path
        model_name=$(basename "$model_path")

        # Construct data path and output path
        data_path="data/${dataset_name}/${split}.jsonl"
        output_path="embeddings/${model_name}/${dataset_name}/${split}"
        
        echo "Processing $data_path with $model_name ($num_layers layers)..."
        python src/get_embeds.py \
            --data_path "$data_path" \
            --output_path "$output_path" \
            --model_path "$model_path" \
            --num_layers "$num_layers"
    done
done
