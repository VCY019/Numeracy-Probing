#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=sbatch_logs/finetune_%j.log
#SBATCH --error=sbatch_logs/finetune_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1

# Dataset configuration
DATA_NAME="int_sci_compare"

# Models with layer counts (format: "model_path:num_layers")
MODEL_CONFIGS=(
    "mistralai/Mistral-7B-v0.1:32"
    "allenai/OLMo-2-1124-7B-Instruct:32"
    "meta-llama/Llama-2-7b-hf:32"
    "Qwen/Qwen3-8B:36"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:28"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B:32"
    "meta-llama/Llama-3.1-8B-Instruct:32"
)

# Finetune hyperparameters
LEARNING_RATE=5e-5
ALPHA=0.1  # regression loss weight
BETA=0.1   # classification loss weight
GAMMA=0.1  # log-ratio loss weight
LAYER_DEPTH=0.9

EPOCHS=3
BATCH_SIZE=16
MAX_SEQ_LENGTH=128
EVALUATE_EVERY=100 # evaluate 8000/16/100 = 5 times per epoch
RANDOM_SEED=42

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

echo "========================================"
echo "FINETUNE"
echo "========================================"
echo "Date: $(date)"
echo "Dataset: ${DATA_NAME}"
echo "Models: ${MODEL_CONFIGS[@]%%:*}"
echo "Learning rate: $LEARNING_RATE"
echo "Alpha=$ALPHA, Beta=$BETA, Gamma=$GAMMA"
echo "Layer depth=$LAYER_DEPTH"
echo "Epochs=$EPOCHS, Batch size=$BATCH_SIZE"
echo "========================================"

# Set data paths
TRAIN_DATA="./data/${DATA_NAME}/train.jsonl"
VAL_DATA="./data/${DATA_NAME}/val.jsonl"

# Loop through all model configurations
for model_config in "${MODEL_CONFIGS[@]}"; do
    IFS=':' read -r model_path num_layers <<< "$model_config"
    model_name=$(basename "$model_path")
    
    # Special case: when layer_depth=1, use the last layer index
    probe_layer=$(python -c "import math; print($num_layers - 1 if $LAYER_DEPTH == 1.0 else math.floor($num_layers * $LAYER_DEPTH))")
    
    echo ""
    echo "---- Model: $model_name ----"
    echo "Total layers: $num_layers"
    echo "Probe layer ($LAYER_DEPTH): $probe_layer"

    exp_name="lr${LEARNING_RATE}_layer${probe_layer}_alpha${ALPHA}_beta${BETA}_gamma${GAMMA}_epochs${EPOCHS}"
    save_dir="./checkpoints/finetune/${DATA_NAME}/${model_name}/${exp_name}"
    logs_dir="./logs/finetune/${DATA_NAME}/${model_name}/${exp_name}"

    mkdir -p "$save_dir" "$logs_dir"

    echo "Running experiment: $exp_name"

    python src/finetune.py \
        --model_name "$model_path" \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --save_path "$save_dir" \
        --logs_path "$logs_dir" \
        --probe_layers "$probe_layer" \
        --num_epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --lm_loss_weight 1.0 \
        --regression_loss_weight "$ALPHA" \
        --classification_loss_weight "$BETA" \
        --regression_diff_loss_weight "$GAMMA" \
        --evaluate_every $EVALUATE_EVERY \
        --max_sequence_length $MAX_SEQ_LENGTH \
        --random_seed $RANDOM_SEED \
        --use_lora \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --use_sklearn_init
        
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $exp_name"
    else
        echo "✗ Failed: $exp_name"
    fi
done

echo "All finetuning jobs completed!" 