# Mean Flow Distillation

![Mean Flow Distillation structure](pics/banner.png)

Official implementation of **"Mean Flow Distillation: Robust and Stable Distillation for Flow Matching Models (ICML 2026)"**.

This repository provides the training code for multi-GPU Mean Flow Distillation (MFD) on **SANA 1.6B**. The current release focuses on prompt-only text-to-image distillation with LoRA adapters built on top of the Hugging Face `diffusers` SANA pipeline.

> 📌 Paper and BibTeX will be added soon.

## ✨ Highlights

- Mean Flow Distillation training for flow matching text-to-image models.
- SANA 1.6B 1024px BF16 backbone support.
- LoRA-based student and auxiliary adapter training.
- Multi-GPU training via `accelerate`.

## 📋 TODO

- [ ] Add paper link.
- [ ] Add pretrained MFD LoRA checkpoints.

## 🧩 Repository Structure

```text
.
├── README.md
├── requirements.txt
└── t2i_SANA
    ├── inference_sana.py          # Example inference script with trained LoRA weights
    ├── train_sana_mfd.sh          # Default multi-GPU launch script
    └── scripts
        └── train_sana_mfd.py      # Main MFD training script
```

## ⚙️ Environment Setup

We recommend using a fresh Conda environment with Python 3.10.

```bash
conda create -n mfd python=3.10
conda activate mfd
```

Install the PyTorch version that matches your CUDA and GPU setup. Please refer to the official [PyTorch installation page](https://pytorch.org/get-started/previous-versions/):

```bash
# Example only. Choose the command that matches your CUDA version.
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

Install the latest `diffusers` from source:

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
cd ..
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The training script checks for `diffusers>=0.36.0.dev0`, so using the source installation above is recommended.

## 📦 Data Preparation

We use **LAION-Aesthetic-6.5+** as the prompt source for distillation. Only the prompt text is used during training.

Download the cleaned prompt file from [here](https://drive.google.com/file/d/1aVVGwTnuw9H-SHmvOkoc9YYx5bWK9KyC/view?usp=share_link) and place it at:

```text
t2i_SANA/datasets/labels_cleaned.tsv
```

The TSV file is expected to contain at least four tab-separated fields per line. The script uses the second field as the prompt.

## 🚀 Training

### Hardware

Minimum:

- 1 GPU with at least 24 GB VRAM

Recommended:

- 8 GPUs with at least 40 GB VRAM each

### Default Training Command

```bash
cd t2i_SANA
bash train_sana_mfd.sh
```

The default script uses:

- Base model: `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers`
- Resolution: `1024`
- LoRA rank / alpha: `64 / 64`
- Batch size per GPU: `1`
- Mixed precision: `bf16`
- Number of GPUs: `8`
- Output directory: `t2i_SANA/outputs`

Edit `t2i_SANA/train_sana_mfd.sh` to change GPU IDs, learning rates, output directory, validation prompt, training length, etc.


Common arguments:

| Argument | Description |
| --- | --- |
| `--prompt_dataset` | Path to the prompt TSV file. |
| `--output_dir` | Directory for LoRA weights, checkpoints, logs, and validation images. |
| `--rank` / `--lora_alpha` | LoRA rank and scaling factor. |
| `--aux_steps_per_cycle` | Number of auxiliary adapter updates per alternating cycle. |
| `--student_steps_per_cycle` | Number of student adapter updates per alternating cycle. |
| `--r_not_equal_t_ratio` | `1.0` for MFD behavior; `0.0` corresponds to the VSD-style setting in the script. |
| `--num_inference_steps` | Number of student inference steps used during validation/training sampling. |
| `--checkpointing_steps` | Save a checkpoint every N optimization steps. |
| `--resume_from_checkpoint` | Resume from a checkpoint path or use `latest`. |
| `--report_to` | Logging backend, e.g. `wandb` or `tensorboard`. |

### Outputs

Training writes files under `--output_dir`, including:

- `checkpoint-*` directories with saved LoRA weights and accelerator state.
- Intermediate decoded samples under `pics/`.

## 🖼️ Inference

After training, update the LoRA checkpoint path in `t2i_SANA/inference_sana.py`:

```python
pipe.load_lora_weights("outputs/checkpoint-1000/pytorch_lora_weights.safetensors")
```

Then run:

```bash
cd t2i_SANA
python inference_sana.py
```

The script loads the SANA 1.6B BF16 pipeline, applies the trained LoRA weights, generates one 1024x1024 image with 4 inference steps, and saves it to:

```text
t2i_SANA/sana.png
```

## 📊 Results

### 4-Step MFD Samples

![4-step Mean Flow Distillation samples](pics/mfd_samples.png)

## 📚 Citation

If you find this repository useful, please consider citing our work.

```bibtex
TODO: Add BibTeX entry after the paper is public.
```

## 🙏 Acknowledgements

This repository builds on and modifies code from the excellent open-source projects below:

- [Hugging Face diffusers](https://github.com/huggingface/diffusers)

We thank the authors and contributors for their valuable work.
