# EdgeECG

This repository contains the implementation of EdgeECG for ECG classification on the MIT-BIH Arrhythmia Database.

## 1. Project Structure

```text
.
├── main.py              # Training and evaluation entry point
├── dataloader.py        # Data loading
├── utils.py             # Utility functions
├── model.py             # Model definition
├── DCP.py               # DCP module
├── ecg_data/mit-bih/    # Place the MIT-BIH dataset here
└── logs/                # Training logs
```

## 2. Environment

- Python 3.10
- Install dependencies with:

```bash
pip install -r requirements.txt
```

## 3. Dataset

Download the MIT-BIH Arrhythmia Database and place it in:

```text
ecg_data/mit-bih/
```

## 4. Running the Code

Place the MIT-BIH dataset in `ecg_data/mit-bih/`, then run:

```bash
python main.py
```

## 5. Output

- `checkpoints/`: Saved trained models
- `logs/`: Training and evaluation logs
