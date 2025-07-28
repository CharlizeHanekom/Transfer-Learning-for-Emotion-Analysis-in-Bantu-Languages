# Emotion Classification for Bantu Languages

This Jupyter notebook trains and evaluates emotion classification models for three African languages: Zulu, Xhosa, and Swahili. The models leverage transfer learning with pre-trained language models, hyperparameter tuning, and gradual unfreezing techniques for efficient training.

## Key Features
- **Multi-language Support**: Trains models for Zulu, Xhosa, and Swahili
- **Model Variety**: Uses three transformer architectures:
  - `AfriBERTa` (castorini/afriberta_large)
  - `XLM-RoBERTa` (xlm-roberta-base)
  - `Serengeti` (UBC-NLP/serengeti-E250)
- **Dataset**: BRIGHTER+ Dataset
- **Hyperparameter Optimization**: Uses Optuna for tuning
- **Gradual Unfreezing**: Implements layer-wise unfreezing during training
- **Performance Metrics**: Tracks F1 scores, ROC AUC, Hamming loss, and more
- **Efficiency Monitoring**: Logs GPU memory, power usage, and inference times

## Notebook Structure
### 1. Environment Setup
   - Installs required packages
   - Configures GPU monitoring
   - Initialises efficiency metrics tracking

### 2. Data Loading & Preprocessing
   - Loads parquet datasets from Google Drive
   - Cleans text data
   - Splits into train/validation/test sets
   - Prepares datasets for model input

### 3. Model Training
   - Hyperparameter tuning with Optuna
   - Gradual unfreezing implementation
   - Custom training loop with efficiency metrics
   - Model saving to Google Drive

### 4. Evaluation
   - Generates predictions for test sets
   - Calculates performance metrics
   - Logs results to Weights & Biases
   - Saves evaluation results to CSV

## How to Use
1. Clone the repository
2. Upload `760Final.ipynb` to Google Colab or run locally with Jupyter
3. Run all cells in sequence

### Usage Requirements
- **Google Drive Mounting**:
  - Dataset paths hardcoded to Google Drive
  - Requires shared drive access
- **Weights & Biases**:
  - Metrics logged to W&B
  - Requires W&B account and API key
- **GPU Acceleration**:
  - Designed for GPU runtime (CUDA)
  - Includes GPU memory monitoring

## Key Outputs
- **Trained Models**: Saved to `/content/drive/Shareddrives/COS 760 Group 13 Project/Models/`
- **Hyperparameters**: Saved to `/content/drive/Shareddrives/COS 760 Group 13 Project/Parameters/`
- **Evaluation Results**: Saved to `/content/drive/Shareddrives/COS 760 Group 13 Project/Results/evaluation_results.csv`

## Contributors
**COS 760 – Group 13 - University of Pretoria (2025)**  
- Charlize Hanekom (`u22487222@tuks.co.za`)
- Jayson du Toit (`u22571532@tuks.co.za`)
- Nonkululeko Ntshele (`u21668452@tuks.co.za`)

## Results Summary
| Model       | Language  | F1 Micro | ROC AUC | Hamming Loss |
|-------------|-----------|----------|---------|--------------|
| AfriBERTa   | isiXhosa  | 0.505    | 0.82    | 0.22         |
| XLM-R       | isiXhosa  | 0.336    | 0.81    | 0.61         |
| Serengeti   | isiXhosa  | 0.492    | 0.83    | 0.25         |
| AfriBERTa   | isiZulu   | 0.002    | 0.71    | 0.08         |
| XLM-R       | Swahili   | 0.187    | 0.56    | 0.74         |

## Files Included
├── 760_BaseModels.ipynb - Baseline model training (no advanced tuning)  
├── 760_Final.ipynb - Full fine-tuning pipeline (layer freezing, metrics)  
├── 760_Group_13_Report.pdf - Final research report  
├── README.md - Project documentation (this file)  
├── evaluation_results.csv  
├── base_models_evaluation_results.csv  

The notebook provides a comprehensive pipeline for training and evaluating emotion classification models on low-resource African languages using state-of-the-art techniques.
