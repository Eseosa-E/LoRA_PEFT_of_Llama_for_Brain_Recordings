# LoRA PEFT of LLaMA for Brain Recordings

A parameter-efficient fine-tuning (PEFT) approach using Low-Rank Adaptation (LoRA) to align a large language model (LLaMA) with fMRI brain recording data, enabling brain-to-language decoding from neural signals.

---

## Overview

This project explores whether a pretrained language model can be fine-tuned — with minimal parameter updates — to decode language from fMRI brain activity. Using LoRA adapters, only a small fraction of model weights are trained, making the approach memory-efficient and suitable for neuroscience datasets where data is typically scarce.

The pipeline takes preprocessed fMRI recordings, projects them into a language model's embedding space, and trains the model to predict the corresponding text stimulus. Results are evaluated on correlation, BLEU, and related metrics.

## Compute

All experiments were run on a single **NVIDIA RTX 3090 (24GB VRAM)**, provisioned via [vast.ai](https://vast.ai). LoRA was central to making this feasible — full fine-tuning of LLaMA at this scale would exceed the available VRAM budget.

---

## Project Structure

\`\`\`
LoRA_PEFT_of_Llama_for_Brain_Recordings/
├── config.py               # Hyperparameters and experiment configuration
├── data.py                 # Dataset loading and preprocessing
├── main.py                 # Training and evaluation entry point
├── model.py                # Core model architecture
├── model_utils.py          # Model helper functions
├── settings.py             # Global settings
├── sub_models.py           # Component submodels (e.g. fMRI encoder)
├── top_model_utils.py      # Top-level model utilities
├── dataset/
│   ├── example.pca1000.wq.pkl   # PCA-reduced fMRI features (1000 components)
│   ├── example.wq.pkl           # Raw/preprocessed fMRI data
│   └── Huth.json                # Stimulus metadata (Huth dataset)
└── results/
    ├── chart1_training_loss.png
    ├── chart2_metrics.png
    ├── chart2_metrics_testset.png
    ├── chart3_predictions_table.png
    ├── chart4_fmri_heatmap.png
    ├── chart5_parameter_efficiency.png
    ├── chart6_memory_comparison.png
    ├── chart7_boxplot_distributions.png
    ├── chart8_correlation_matrix.png
    ├── chart9_scatter_length_vs_score.png
    ├── chart10_fmri_pca_correlation.png
    ├── chart11_loss_convergence_annotated.png
    ├── chart12_comprehensive_summary.png
    ├── id2info.json
    └── info.json

\`\`\`

---

## Dataset

This project uses fMRI data from the **Huth et al.** natural language listening paradigm, where participants listened to spoken narratives while brain activity was recorded. The `.pkl` files contain preprocessed and PCA-reduced BOLD signals aligned to word-level stimuli.

---

## Results

Training and evaluation outputs are saved to `results/`. This includes loss curves, prediction samples, fMRI activation heatmaps, and efficiency comparisons between full fine-tuning and LoRA. Checkpoints are saved per epoch as `.pt` files.

---

## Dependencies

Key libraries used: `transformers`, `peft`, `torch`, `numpy`, `scikit-learn`, `matplotlib`.

Install manually or run:
\`\`\`bash
pip install transformers peft torch numpy scikit-learn matplotlib
\`\`\`

---

## Acknowledgements

fMRI data sourced from the Huth et al. dataset. LLaMA model weights accessed via HuggingFace. LoRA implementation via the `peft` library by HuggingFace.