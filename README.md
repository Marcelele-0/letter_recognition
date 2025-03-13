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
