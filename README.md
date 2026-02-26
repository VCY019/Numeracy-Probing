# Numeracy Probing

Code and data to reproduce experiments in the paper [LLMs Know More About Numbers than They Can Say](https://arxiv.org/abs/2602.07812) (EACL 2026, Oral)

## Setup

```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── src/                        # Core Python scripts
│   ├── construct_data.py       # Generate synthetic comparison datasets
│   ├── get_embeds.py           # Extract embeddings (synthetic data)
│   ├── get_embeds_arxiv.py     # Extract embeddings (arXiv data)
│   ├── train_probe.py          # Train and evaluate probes
│   ├── train_probe_arxiv.py    # Train and evaluate probes on arXiv embeddings
│   ├── verbalization.py        # Test LLM verbalization accuracy
│   └── finetune.py             # Probe-aware fine-tuning with LoRA
├── scripts/                    # Shell scripts to run the pipeline
│   ├── construct_data.sh
│   ├── get_embeds.sh
│   ├── get_embeds_arxiv.sh
│   ├── train_probe.sh
│   ├── train_probe_arxiv.sh
│   ├── verbalization.sh
│   ├── finetune.sh             # Example fine-tuning script
│   └── verbalization-analysis/ # Few-shot and alt-prompt analysis scripts
├── notebooks/                  # Figure generation and error analysis
├── data/                       # Generated/downloaded datasets
├── verbalization-test/         # Verbalization results (including GPT-4.1 and GPT-4.1-mini results)
├── gpt-experiments/            # Scripts to reproduce GPT-4.1 and GPT-4.1-mini results
└── figures/                    # Generated figures
```

## Pipeline

All scripts assume execution from the **repository root**. All shell scripts include SLURM headers that can be ignored if running locally.

### Step 1: Construct Synthetic Data

```bash
bash scripts/construct_data.sh
```

Generates `int_sci_compare` and `dec_sci_compare` datasets (8000 train / 1600 val / 1600 test each) under `data/`.

### Step 2: Download ArXiv Data

Download `arxiv_100k.jsonl` from [HuggingFace](https://huggingface.co/datasets/VCY019/Numeracy-Probing) and place it at `data/arxiv_100k.jsonl`. This file is too large for GitHub.

### Step 3: Extract Embeddings

```bash
bash scripts/get_embeds.sh          # Synthetic data (all models × all splits)
bash scripts/get_embeds_arxiv.sh    # ArXiv data (Mistral-7B only)
```

### Step 4: Train Probes

```bash
bash scripts/train_probe.sh         # Synthetic probes (regression, classification, log-ratio regression)
bash scripts/train_probe_arxiv.sh   # ArXiv probes (regression only)
```

### Step 5: Verbalization

```bash
bash scripts/verbalization.sh       # Standard 1-shot verbalization on test set
```

For additional verbalization analyses (few-shot sweeping, alternative prompts):

```bash
bash scripts/verbalization-analysis/verbalization-{1,2,3,4,5}-shot.sh
bash scripts/verbalization-analysis/verbalization-alt.sh
```

### Step 6: Fine-tuning

`scripts/finetune.sh` provides an example configuration. Adapt it for hyperparameter sweeping as needed.

> **Note:** Fine-tuning requires an **80GB GPU** (40GB is insufficient).

After fine-tuning, the finetuned models should go through the verbalization process (Step 5) using **both val and test** data. Use the accuracy on **val** to select the best hyperparameters, and report the verbalization accuracy on **test**.

### Step 7: GPT-4.1 Results

To reproduce the GPT-4.1 and GPT-4.1-mini results in `verbalization-test/`, run from the `gpt-experiments/` directory with `OPENAI_API_KEY` set:

```bash
cd gpt-experiments
bash gpt_verbalization.sh
```

The script outputs use lowercase model names (e.g. `gpt-4.1`), but the committed files in `verbalization-test/` use uppercase (`GPT-4.1`). The naming convention otherwise follows:

```
verbalization-test/{setting}_output_{model}.txt          # One-shot (e.g., int_sci_compare_output_GPT-4.1.txt)
verbalization-test/{setting}_output_{model}_alt.txt      # One-shot with numbers reversed in example
verbalization-test/{setting}_output_{model}_{n}shot.txt  # Few-shot (n > 1)
```

### Step 8: Reproduce results

Run notebooks in `notebooks/` to reproduce paper figures and statistics. Note that probe scatter plots (Figure 1) are automatically created during Step 4 when running `src/train_probe.py` (with `--eval_test`) and `src/train_probe_arxiv.py`. All other figures are saved to `figures/`.

#### Script Mapping

| Output | Source |
|--------|--------|
| Figure 1 | `src/train_probe.py`, `src/train_probe_arxiv.py` |
| Figures 2, 4, 5, 6, 7, 8, 9 | `notebooks/draw_figures.ipynb` |
| Figures 3, 10, 13, 14 | `notebooks/error_analysis_probes.ipynb` |
| Figure 11 | `notebooks/error_analysis_alt_prompt.ipynb` |
| Figure 12 | `notebooks/error_analysis_few_shot.ipynb` |
| Relative errors in abstract | `notebooks/abstract_regression_error_synthetic.ipynb`, `notebooks/abstract_regression_error_arxiv.ipynb` |

## Models

| Model | Layers |
|-------|--------|
| DeepSeek-R1-Distill-Llama-8B | 32 |
| DeepSeek-R1-Distill-Qwen-7B | 28 |
| Llama-2-7b-hf | 32 |
| Llama-3.1-8B-Instruct | 32 |
| Mistral-7B-v0.1 | 32 |
| OLMo-2-1124-7B-Instruct | 32 |
| Qwen3-8B | 36 |
| GPT-4.1 | — |
| GPT-4.1-mini | — |

## Citation

```bibtex
@inproceedings{yuchi-du-eisner-2026,
  author =      {Fengting Yuchi and Li Du and Jason Eisner},
  title =       {{LLM}s Know More About Numbers than They Can Say},
  booktitle =   {Proceedings of the Conference of the European Chapter
                 of the Association for Computational Linguistics: Human
                 Language Technologies (EACL)},
  year =        {2026},
  month =       mar,
  address =     {Rabat, Morocco},
  note =        {Oral presentation.},
  url =         {https://arxiv.org/abs/2602.07812}
}
```