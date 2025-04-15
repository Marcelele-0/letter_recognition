# Letter Recognition - Conda Environment

## Creating the Environment

To create a Conda environment from the `env.yaml` file, follow these steps:

1. Make sure you have Conda installed.
2. Open a terminal and navigate to the directory containing the `env.yaml` file.
3. Run the command:
   ```bash
   conda env create -f environment.yaml
   ```
4. After the installation is complete, activate the environment:
   ```bash
   conda activate letter_recognition
   ```

## Removing the Environment

If you want to remove the environment, use the command:
```bash
conda remove --name letter_recognition --all
```

## 2. Training Conclusions
###  Model Performance Summary (MNIST)

| Metric                  | Value  | Description |
|----------------------|--------|-------------|
| **Train Accuracy** (`train/acc`) | **87.95%** | Percentage of correctly classified samples in the training set. |
| **Validation Accuracy** (`val/acc`) | **94.80%** | Accuracy on the validation set, indicating generalization performance. |
| **Test Accuracy** (`test/acc`) | **98.05%** | Final model accuracy on the test set. |
| **Train Precision** (`train/precision`) | **88.10%** | Precision on the training set – percentage of correctly predicted positive cases. |
| **Validation Precision** (`val/precision`) | **95.03%** | Precision on the validation set – measures false positive rate. |
| **Test Precision** (`test/precision`) | **98.12%** | Precision on the test set – macro-averaged across all classes. |
| **Train Recall** (`train/recall`) | **87.98%** | Recall on the training set – percentage of actual positives correctly identified. |
| **Validation Recall** (`val/recall`) | **94.77%** | Recall on the validation set – measures false negative rate. |
| **Test Recall** (`test/recall`) | **98.06%** | Recall on the test set – macro-averaged across all classes. |
| **Train Loss** (`train/loss`) | **0.519** | Training loss – lower values indicate better fit to training data. |
| **Validation Loss** (`val/loss`) | **0.331** | Validation loss – helps detect overfitting. |
| **Test Loss** (`test/loss`) | **0.138** | Final loss on the test set – lower is better. |
| **Best Validation Accuracy** (`val/acc_best`) | **94.80%** | The best validation accuracy achieved during training. |

###  Key Insights

**Strong generalization** – The model achieves **93.7% accuracy** on both validation and test sets.  
**Balanced Precision & Recall** – The model does not overly favor any class.  
**Train Accuracy (86.6%) is lower than Validation/Test Accuracy (~93.7%)**, which suggests:  
   - Strong regularization (`weight_decay` or dropout effects).  
   - Differences in data augmentation between training and validation/test sets.  
   - Possible underfitting – further training might improve performance.  


## Project Structure

```
number_recognition/
├── config/
│   ├── logger/
│   │   └── wandb.yaml          # Configuration for W&B logger
│   ├── paths/
│   │   └── deafult.yaml        # Paths configuration (data, logs, output)
│   └── train.yaml              # (Not provided, likely contains training configurations)
├── data/
│   ├── mnist_datamodule.py     # PyTorch Lightning DataModule for MNIST
│   └── my_data/                # Directory for custom data (used in XAI script)
├── models/
│   ├── lenet.py                # Implementation of LeNet-5 model
│   ├── mnist_module.py         # Lightning module for MNIST training
│   └── __init__.py             # (Optional, for module initialization)
├── src/
│   ├── train.py                # Main training script using Hydra and PyTorch Lightning
│   ├── xai_explain.py          # XAI script for explaining predictions with saliency maps
│   └── utils/
│       └── split_data.py       # Utility for calculating train/val/test splits
├── wandb_logs/                 # Directory for W&B logs and checkpoints
│   └── MNIST-Training/
│       └── h57gb7f6/
│           └── checkpoints/    # Checkpoints saved during training
│               └── epoch=9-step=8440.ckpt
├── README.md                   # Documentation for the project
└── environment.yaml            # Conda environment configuration file
```

This structure provides an overview of the project's organization and key components.