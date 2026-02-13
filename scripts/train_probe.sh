#!/bin/bash
#SBATCH --job-name=train_probe
#SBATCH --output=sbatch_logs/train_probe_%j.log
#SBATCH --error=sbatch_logs/train_probe_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=192000

# Data name
DATA_NAME=(
    "int_sci_compare"
    "dec_sci_compare"
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
    data_dir="data/${data_name}"
    
    for model_config in "${MODEL_CONFIGS[@]}"; do
        # Split model config into model_name and num_layers
        IFS=':' read -r model_path num_layers <<< "$model_config"
        
        # Extract model name for paths (remove organization prefix)
        model_name=$(basename "$model_path")
        
        embed_dir="embeddings/${model_name}/${data_name}"
        
        # Check if embeddings exist
        if [ ! -d "$embed_dir" ]; then
            echo "Skipping $model_name on $data_name - embeddings not found at $embed_dir"
            continue
        fi
        
        echo "Training probes for $data_name with $model_name ($num_layers layers)..."
        
        # Run the training script
        python src/train_probe.py \
            --data_dir "$data_dir" \
            --embed_dir "$embed_dir" \
            --num_layers "$num_layers" \
            --model_name "$model_name" \
            --probe_types "regression" "classification" "regression_diff" \
            --eval_test \
            --cross_notation_eval \
        
        echo "Completed training for $model_name on $data_name"
    done
done

echo "All probe training jobs completed!" 