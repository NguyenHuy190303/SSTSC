
## Requirements

- Python 3.6 or 3.7

- torchvision version 0.5.0

## Setup

### Step 1: Create Conda Environment

Run the following command to create a conda environment with the appropriate Python version:

```sh
conda create -n myenv python=3.7 -y
```

### Step 2: Activate Conda Environment

Activate the conda environment:

```sh
conda activate myenv
```

### Step 3: Install Dependencies

Install scikit-learn, torch, and torchvision:

```sh
pip install scikit-learn torch==1.4.0 torchvision==0.5.0
```

### Step 4: Verify Installation

Verify the installation by running:

```sh
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Run Model Training and Evaluation

### Semi-Supervised Training and Test

Run the following command for semi-supervised training and testing:

```bash
python train_ssl.py --dataset_name CricketX --model_name train_SemiInterPF --alpha 0.3 --label_ratio [0.1 0.2 0.4 1.0]
```

### Supervised Training and Test

Run the following command for supervised training and testing:

```bash
python train_ssl.py --dataset_name CricketX --model_name SupCE --label_ratio [0.1 0.2 0.4 1.0]
```

## Check Results

After running model training and evaluation, the checkpoints of the trained model are saved in the local `ckpt` directory, the training logs are saved in the local `log` directory, and all experimental results are saved in the local `results` directory.

