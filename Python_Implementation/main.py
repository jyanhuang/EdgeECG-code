import os
import datetime
import time
import random
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from dataloader import load_all_data, print_split_counts
from DCP import prune
from utils import  plot_heat_map_row_normalized, model_select, compute_fold_metrics

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

save_root = "./logs/logs_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(save_root, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("Using {} device".format(device))

def quantize_conv(weight, bias):
    weight = nn.Parameter(torch.round(weight * (2 ** 10)).to(torch.int16), requires_grad=False)
    bias = nn.Parameter(torch.round(bias * (2 ** 10)).to(torch.int16), requires_grad=False)
    return weight, bias

def dequantize_conv(weight, bias):
    weight = nn.Parameter((weight / (2 ** 10)).to(torch.float32), requires_grad=True)
    bias = nn.Parameter((bias / (2 ** 10)).to(torch.float32), requires_grad=True)
    return weight, bias

class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)   # float32
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)


def train_steps(loop, model, criterion, optimizer, model_name, quantize=False):
    train_loss = []
    train_acc = []
    model.train()

    for step_index, (X, y) in loop:
        if quantize:
            model.conv1.weight, model.conv1.bias = quantize_conv(model.conv1.weight, model.conv1.bias)

        X, y = X.to(device), y.to(device)

        if model_name == 'Ours':
            pred = model(X, quantize)
        else:
            pred = model(X)

        loss = criterion(pred, y)

        if quantize:
            model.conv1.weight, model.conv1.bias = dequantize_conv(model.conv1.weight, model.conv1.bias)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        train_loss.append(loss_value)

        pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        acc = accuracy_score(y_np, pred_result)
        train_acc.append(acc)
        loop.set_postfix(loss=loss_value, acc=acc)

    return {"loss": np.mean(train_loss), "acc": np.mean(train_acc)}


def test_steps(loop, model, criterion, model_name, quantize=False):
    test_loss = []
    test_acc = []
    all_y_true = []
    all_y_pred = []

    model.eval()
    with torch.no_grad():
        if quantize:
            model.conv1.weight, model.conv1.bias = quantize_conv(model.conv1.weight, model.conv1.bias)

        for step_index, (X, y) in loop:
            X, y = X.to(device), y.to(device)

            if model_name == 'Ours':
                pred = model(X, quantize)
            else:
                pred = model(X)

            loss = criterion(pred, y).item()
            test_loss.append(loss)

            pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            acc = accuracy_score(y_np, pred_result)
            test_acc.append(acc)

            all_y_true.extend(y_np.tolist())
            all_y_pred.extend(pred_result.tolist())

            loop.set_postfix(loss=loss, acc=acc)

        if quantize:
            model.conv1.weight, model.conv1.bias = dequantize_conv(model.conv1.weight, model.conv1.bias)

    return {
        "loss": np.mean(test_loss),
        "acc": np.mean(test_acc),
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred)
    }


def train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer, fold_id, quantize=False):
    num_epochs = config['num_epochs']
    train_loss_ls = []
    train_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []

    best_acc = -1
    best_y_true = None
    best_y_pred = None
    best_model_path = os.path.join(save_root, f'best_model_fold_{fold_id}.pt')

    for epoch in range(num_epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

        train_loop.set_description(f'Fold [{fold_id}] Epoch [{epoch + 1}/{num_epochs}]')
        test_loop.set_description(f'Fold [{fold_id}] Epoch [{epoch + 1}/{num_epochs}]')

        train_metrix = train_steps(train_loop, model, criterion, optimizer, config['model_name'], quantize)
        test_metrix = test_steps(test_loop, model, criterion, config['model_name'], quantize)

        train_loss_ls.append(train_metrix['loss'])
        train_acc_ls.append(train_metrix['acc'])
        test_loss_ls.append(test_metrix['loss'])
        test_acc_ls.append(test_metrix['acc'])

        print(f'Fold {fold_id} Epoch {epoch + 1}: train loss={train_metrix["loss"]:.6f}, train acc={train_metrix["acc"]:.6f}')
        print(f'Fold {fold_id} Epoch {epoch + 1}: test  loss={test_metrix["loss"]:.6f}, test  acc={test_metrix["acc"]:.6f}')

        writer.add_scalar(f'fold_{fold_id}/train/loss', train_metrix['loss'], epoch)
        writer.add_scalar(f'fold_{fold_id}/train/accuracy', train_metrix['acc'], epoch)
        writer.add_scalar(f'fold_{fold_id}/validation/loss', test_metrix['loss'], epoch)
        writer.add_scalar(f'fold_{fold_id}/validation/accuracy', test_metrix['acc'], epoch)

        if test_metrix['acc'] > best_acc:
            best_acc = test_metrix['acc']
            best_y_true = test_metrix['y_true'].copy()
            best_y_pred = test_metrix['y_pred'].copy()
            torch.save(model.state_dict(), best_model_path)

    history = {
        'train_loss': train_loss_ls,
        'train_acc': train_acc_ls,
        'test_loss': test_loss_ls,
        'test_acc': test_acc_ls
    }

    return history, best_y_true, best_y_pred, best_model_path


def save_fold_results_csv(all_fold_metrics, save_path):
    fieldnames = ['Fold', 'Accuracy', 'Precision', 'Recall', 'F1N', 'F1S', 'F1V', 'F1F', 'F1Q', 'Favg']
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_fold_metrics:
            writer.writerow(row)


def main():
    config = {
        'seed': 42,
        'num_epochs': 30,
        'batch_size': 128,
        'lr': 0.001,
        'prune': True,
        'model_name': 'Ours',   # LightX3ECG, ecgTransForm, MSDNN, CMM, Ours
        'mode': 0,              # 0: 你的剪枝算法, 1: 按大小剪枝, 2: 随机剪枝
        'quantize': False,
        'num_folds': 5
    }

    quantize = config['quantize']

    X, y = load_all_data()

    if quantize:
        X = np.round(X * (2 ** 8)).astype(np.int16)

    skf = StratifiedKFold(
        n_splits=config['num_folds'],
        shuffle=True,
        random_state=config['seed']
    )

    all_fold_metrics = []

    for fold_id, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print("=" * 80)
        print(f"Start Fold {fold_id}")
        print("=" * 80)

        fold_log_dir = os.path.join(save_root, f'fold_{fold_id}_tb')
        writer = SummaryWriter(log_dir=fold_log_dir)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f"\nFold {fold_id} sample counts:")
        print_split_counts(y_train, y_test)

        train_dataset = ECGDataset(X_train, y_train)
        test_dataset = ECGDataset(X_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        model = model_select(config['model_name']).to(device)

        if config['prune']:
            prune(model, 'conv1', 0.7, 0.8, config['mode'])
            prune(model, 'fc1.0', 0.7, 0.5, config['mode'])
            prune(model, 'fc2', 0.7, 0.3, config['mode'])

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        history, best_y_true, best_y_pred, best_model_path = train_epochs(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            writer=writer,
            fold_id=fold_id,
            quantize=quantize
        )
        writer.close()

        plot_heat_map_row_normalized(best_y_true, best_y_pred, fold_id, save_dir=save_root)

        fold_metrics = compute_fold_metrics(best_y_true, best_y_pred)
        row = {'Fold': fold_id}
        row.update(fold_metrics)
        all_fold_metrics.append(row)

        print(f"\nFold {fold_id} metrics:")
        for k, v in fold_metrics.items():
            print(f"{k}: {v:.4f}")

        # 统计该折 best model 的推理耗时、参数量、FLOPs
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        y_pred = []
        start_time = time.time()
        total_test = 1

        if quantize:
            model.conv1.weight, model.conv1.bias = quantize_conv(model.conv1.weight, model.conv1.bias)

        for _ in range(total_test):
            with torch.no_grad():
                for step_index, (X_batch, y_batch) in enumerate(test_dataloader):
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    if config['model_name'] == 'Ours':
                        pred = model(X_batch)
                    else:
                        pred = model(X_batch)
                    pred_result = torch.argmax(pred, dim=1).cpu().numpy()
                    y_pred.extend(pred_result)

        if quantize:
            model.conv1.weight, model.conv1.bias = dequantize_conv(model.conv1.weight, model.conv1.bias)

        end_time = time.time()
        inference_time = (end_time - start_time) / total_test
        print(f'Fold {fold_id} Inference Time: {inference_time:.6f} seconds')

        with torch.no_grad():
            flops, params = profile(model, inputs=(torch.ones((128, 300)).to(device),))
            print(f"Fold {fold_id} Total parameters: {params}")
            print(f"Fold {fold_id} Total FLOPs: {flops}")

    save_fold_results_csv(all_fold_metrics, os.path.join(save_root, 'five_fold_metrics.csv'))

    print("\n" + "=" * 80)
    print("5-Fold Results")
    print("=" * 80)

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1N', 'F1S', 'F1V', 'F1F', 'F1Q', 'Favg']
    for row in all_fold_metrics:
        print(
            f"Fold {row['Fold']}: "
            + " | ".join([f"{m}={row[m]:.4f}" for m in metric_names])
        )

    print("\nMean ± Std")
    for m in metric_names:
        values = [row[m] for row in all_fold_metrics]
        print(f"{m}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == '__main__':
    main()