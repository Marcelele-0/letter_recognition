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
| **Train Accuracy** (`train/acc`) | **86.66%** | Percentage of correctly classified samples in the training set. |
| **Validation Accuracy** (`val/acc`) | **93.77%** | Accuracy on the validation set, indicating generalization performance. |
| **Test Accuracy** (`test/acc`) | **93.74%** | Final model accuracy on the test set. |
| **Train Precision** (`train/precision`) | **87.30%** | Precision on the training set – percentage of correctly predicted positive cases. |
| **Validation Precision** (`val/precision`) | **93.49%** | Precision on the validation set – measures false positive rate. |
| **Test Precision** (`test/precision`) | **93.68%** | Precision on the test set – macro-averaged across all classes. |
| **Train Recall** (`train/recall`) | **86.53%** | Recall on the training set – percentage of actual positives correctly identified. |
| **Validation Recall** (`val/recall`) | **93.65%** | Recall on the validation set – measures false negative rate. |
| **Test Recall** (`test/recall`) | **93.60%** | Recall on the test set – macro-averaged across all classes. |
| **Train Loss** (`train/loss`) | **0.591** | Training loss – lower values indicate better fit to training data. |
| **Validation Loss** (`val/loss`) | **0.395** | Validation loss – helps detect overfitting. |
| **Test Loss** (`test/loss`) | **0.404** | Final loss on the test set – lower is better. |
| **Best Validation Accuracy** (`val/acc_best`) | **93.77%** | The best validation accuracy achieved during training. |

###  Key Insights

**Strong generalization** – The model achieves **93.7% accuracy** on both validation and test sets.  
**Balanced Precision & Recall** – The model does not overly favor any class.  
**Train Accuracy (86.6%) is lower than Validation/Test Accuracy (~93.7%)**, which suggests:  
   - Strong regularization (`weight_decay` or dropout effects).  
   - Differences in data augmentation between training and validation/test sets.  
   - Possible underfitting – further training might improve performance.  


