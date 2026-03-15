# DeepDESmelt

**A Composition-Aware Mixture-of-Experts Model for Melting Point Prediction of Deep Eutectic Solvents**

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Overview

DeepDESmelt is a deep learning framework for predicting the melting points of Deep Eutectic Solvents (DES). It employs a Mixture-of-Experts (MOE) architecture with composition-aware feature learning to capture the complex relationships between molecular structures, molar ratios, and melting points.

### Key Features

- **Composition-Aware Learning**: FiLM (Feature-wise Linear Modulation) for molar ratio conditioning
- **Mixture-of-Experts Architecture**: Multiple expert networks with dynamic gating mechanism
- **UniMol Representations**: State-of-the-art molecular representations from UniMol
- **Strict Data Splitting Strategy**: Group-based splitting ensures no HBA-HBD pair leakage between train/val/test sets
- **Easy-to-Use CLI**: Unified command-line interface for inference and data preparation

### Data Splitting Strategy

DeepDESmelt uses a **strict group-based splitting strategy** to prevent data leakage:

- **Grouping Rule**: All samples with the same HBA-HBD pair are grouped together, regardless of molar ratio
- **Split Guarantee**: Each HBA-HBD pair appears in **only one** of train/val/test sets
- **Example**: If (ChCl, Urea) appears in the training set with molar ratio 1:2, then (ChCl, Urea) with any other molar ratio (e.g., 1:1, 1:3) will also be in the training set, never in validation or test sets
- **Benefit**: This ensures the model's ability to generalize to completely new HBA-HBD combinations, not just new molar ratios of seen pairs

This splitting strategy is implemented using `GroupShuffleSplit` from scikit-learn, providing a rigorous evaluation of the model's generalization capability.

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/buyudaren3/DeepDESmelt.git
cd DeepDESmelt
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate deepdesmelt
```

3. **(Optional) Download reference model weights**:

   If you want to use our reference model directly without training from scratch:
   
   - Download from [GitHub Releases](https://github.com/buyudaren3/DeepDESmelt/releases)
   
   Place the downloaded `best_model.pth` in:
   ```bash
   mkdir -p checkpoints/1/
   # Move the downloaded file to checkpoints/1/best_model.pth
   ```

### Basic Usage

#### Option 1: Use Our Reference Model (Recommended for Quick Start)

1. **Generate Dataset for Your Data**:
```bash
python main.py generate_dataset \
    --mode inference \
    --train_data_path ./processed_data/DES_melting_point_dataset/ \
    --data_file ./raw_data/your_data.csv \
    --output_name your_data \
    --save_path ./inference_output/
```

2. **Make Predictions**:
```bash
python main.py predict \
    --model_path ./checkpoints/1/best_model.pth \
    --norm_file ./processed_data/DES_melting_point_dataset/data_range.npz \
    --hidden_size 256 \
    --num_hidden_layers 1 \
    --num_experts 4 \
    --norm LayerNorm \
    --activation LeakyReLU \
    --data_file ./inference_output/your_data.pt \
    --output_file ./results/predictions.csv
```

### 📌 Important Notes

#### Normalization Parameters Consistency

The `--norm_file` in `predict` command must match the `--train_data_path` used in `generate_dataset`:

**Example Workflow**:
```bash
# Step 1: Generate inference data using training dataset A
python main.py generate_dataset \
    --mode inference \
    --train_data_path ./processed_data/dataset_A/ \
    --data_file ./raw_data/new_data.csv \
    --save_path ./inference_output/

# Step 2: Predict using the SAME dataset A's normalization
python main.py predict \
    --norm_file ./processed_data/dataset_A/data_range.npz \
    --data_file ./inference_output/new_data.pt \
    --output_file ./results.csv
```

**Why It Matters**: Each training dataset has its own normalization parameters (min/max values). Using mismatched normalization files will cause incorrect prediction.

#### Model Hyperparameters

When using the `predict` command, provide the same hyperparameters as the trained model (`--hidden_size`, `--num_hidden_layers`, `--norm`, `--activation`, `--num_experts`). These define the model architecture and must match exactly.

## Reproducing Paper Results

We provide complete materials for reproducing all results in our paper.

> **Note on Data Splitting**: Our dataset uses a strict group-based splitting strategy where each HBA-HBD pair appears in only one of train/val/test sets. This ensures rigorous evaluation of the model's ability to generalize to completely new molecular combinations.

### Quick Reproduction

Use our reference model and preprocessed data to verify paper results:

```bash
# Run prediction on test set
python main.py predict \
    --model_path ./checkpoints/1/best_model.pth \
    --data_file ./processed_data/DES_melting_point_dataset/test.pt \
    --output_file ./results/test_predictions.csv
```

**Expected Results**: R² = 0.931, MAE = 12.2 K, RMSE = 17.0 K, AARD = 3.7%

**Provided Materials**:
- Reference model: `checkpoints/1/best_model.pth`
- Original dataset: `raw_data/DES_melting_point_dataset.csv`
- Preprocessed features: `processed_data/DES_melting_point_dataset/*.pt`

### Reproducibility Notes

**Same Machine**: Training is fully deterministic on the same hardware. Multiple runs with the same seed produce identical results.

**Cross-Machine**: Minor variations (~2-3% in R²) may occur across different hardware due to GPU-specific implementations. This is expected in deep learning.

**Inference**: Fully reproducible across all machines using provided weights.

## 📁 Project Structure

```
DeepDESmelt/
├── deepdesmelt/                   # Main Python package
│   ├── __init__.py
│   ├── architecture/              # Model architecture
│   │   └── DeepDESmelt.py         # MOE model definition
│   ├── data/                      # Data processing
│   │   └── preprocessing.py       # Dataset generation (UniMol feature extraction)
│   ├── inference/                 # Inference utilities
│   │   └── predictor.py           # Inference script
│   └── utils/                     # Common utilities
│       └── visualization.py
│       
├── checkpoints/                   # Reference model weights (provided by author)
│   └── 1/best_model.pth
├── raw_data/                      # Input dataset files (CSV format)
│   └── DES_melting_point_dataset.csv
├── processed_data/                # Processed datasets (.pt files)
│   └── DES_melting_point_dataset/ # Use to quickly generate dataset for your data
├── main.py                        # Main entry point
├── args.py                        # Argument parser
├── environment.yml                # Conda environment
├── README.md                      # This file
└── .gitignore                     # Git ignore rules

# Runtime directories (created automatically, not in repo):
# ├── results/                     # Prediction results (.csv files)
# └── inference_output/            # Inference outputs
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [luoguang3333@163.com](mailto:luoguang3333@163.com).

## 🙏 Acknowledgments

- DeepModeling team for the molecular representation model Uni-Mol2
