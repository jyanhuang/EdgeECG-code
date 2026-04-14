import torch
import wfdb
import pywt
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
# from compare.ECGTransForm import ecgTransForm
# from compare.MSDNN import MSDNN
# from compare.LightX3ECG.nets import LightX3ECG
# from compare.CMM.CMM import MyNet
from net import Model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# AAMI beat-level mapping
# =========================
AAMI_LABEL_MAP = {
    # N: Non-ectopic beats
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',

    # S: Supraventricular ectopic beats
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',

    # V: Ventricular ectopic beats
    'V': 'V', 'E': 'V',

    # F: Fusion beats
    'F': 'F',

    # Q: Unknown / paced / unclassifiable beats
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']
AAMI_CLASS_TO_INDEX = {c: i for i, c in enumerate(AAMI_CLASSES)}
INDEX_TO_AAMI_CLASS = {i: c for c, i in AAMI_CLASS_TO_INDEX.items()}

def print_split_counts(y_train, y_test, class_names=AAMI_CLASSES):
    train_counter = Counter(y_train.tolist() if hasattr(y_train, 'tolist') else y_train)
    test_counter = Counter(y_test.tolist() if hasattr(y_test, 'tolist') else y_test)

    print("\n{:<10}{:<12}{:<12}{:<12}".format("Class", "Train set", "Test set", "Total"))
    print("-" * 46)

    total_train = 0
    total_test = 0
    total_all = 0

    for i, cls in enumerate(class_names):
        train_num = train_counter.get(i, 0)
        test_num = test_counter.get(i, 0)
        total_num = train_num + test_num

        total_train += train_num
        total_test += test_num
        total_all += total_num

        print("{:<10}{:<12}{:<12}{:<12}".format(cls, train_num, test_num, total_num))

    print("-" * 46)
    print("{:<10}{:<12}{:<12}{:<12}".format("Total", total_train, total_test, total_all))
    print()

# wavelet denoise preprocess using mallat algorithm
def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)

    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def beat_to_aami(symbol):
    """
    Convert original MIT-BIH annotation symbol to AAMI class index.
    Return None if the symbol is not used in AAMI mapping.
    """
    if symbol not in AAMI_LABEL_MAP:
        return None
    aami_label = AAMI_LABEL_MAP[symbol]
    return AAMI_CLASS_TO_INDEX[aami_label]


# load the ecg data and the corresponding labels, then denoise the data using wavelet transform
def get_data_set(number, X_data, Y_data, win_left=99, win_right=201):
    print(f"loading the ecg data of No.{number}")

    # load ECG signal
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    # load annotations
    annotation = wfdb.rdann('ecg_data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # remove unstable data at the beginning and the end
    start = 10
    end = 5
    i = start
    j = len(Rclass) - end

    while i < j:
        symbol = Rclass[i]
        label = beat_to_aami(symbol)

        # skip symbols not included in AAMI mapping
        if label is None:
            i += 1
            continue

        # boundary check
        left = Rlocation[i] - win_left
        right = Rlocation[i] + win_right
        if left < 0 or right > len(rdata):
            i += 1
            continue

        x_train = rdata[left:right]

        # ensure fixed length = 300
        if len(x_train) != (win_left + win_right):
            i += 1
            continue

        X_data.append(x_train)
        Y_data.append(label)
        i += 1


# load dataset and preprocess
def load_data(ratio=0.2, random_seed=42):
    numberSet = [
        '100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
        '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
        '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
        '231', '232', '233', '234'
    ]

    dataSet = []
    labelSet = []

    for n in numberSet:
        get_data_set(n, dataSet, labelSet)

    dataSet = np.array(dataSet, dtype=np.float32).reshape(-1, 300)
    labelSet = np.array(labelSet, dtype=np.int64).reshape(-1)

    print("Dataset shape:", dataSet.shape)
    print("Label shape:", labelSet.shape)
    print("AAMI class distribution:")
    for idx, cls_name in enumerate(AAMI_CLASSES):
        print(f"  {cls_name}: {(labelSet == idx).sum()}")

    # stratified split is strongly recommended for imbalanced ECG classes
    X_train, X_test, y_train, y_test = train_test_split(
        dataSet,
        labelSet,
        test_size=ratio,
        random_state=random_seed,
        stratify=labelSet
    )
    return X_train, X_test, y_train, y_test


def load_all_data():
    numberSet = [
        '100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
        '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
        '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
        '231', '232', '233', '234'
    ]

    dataSet = []
    labelSet = []

    for n in numberSet:
        get_data_set(n, dataSet, labelSet)

    X = np.array(dataSet, dtype=np.float32).reshape(-1, 300)
    y = np.array(labelSet, dtype=np.int64).reshape(-1)

    print("Total dataset shape:", X.shape)
    print("Total label shape:", y.shape)
    for i, c in enumerate(AAMI_CLASSES):
        print(f"{c}: {(y == i).sum()}")

    return X, y

def compute_fold_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    per_class_f1 = f1_score(
        y_true, y_pred,
        average=None,
        labels=[0, 1, 2, 3, 4],
        zero_division=0
    )

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1N": per_class_f1[0],
        "F1S": per_class_f1[1],
        "F1V": per_class_f1[2],
        "F1F": per_class_f1[3],
        "F1Q": per_class_f1[4],
        "Favg": np.mean(per_class_f1)
    }
    return metrics

# confusion matrix
def plot_heat_map(y_test, y_pred, save_path='confusion_matrix_aami.png'):
    con_mat = confusion_matrix(y_test, y_pred, labels=list(range(len(AAMI_CLASSES))))

    plt.figure(figsize=(8, 8))
    seaborn.heatmap(
        con_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=AAMI_CLASSES,
        yticklabels=AAMI_CLASSES
    )
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (AAMI)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_history_tf(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy_aami.png', dpi=300)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss_aami.png', dpi=300)
    plt.show()


def plot_history_torch(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history['train_acc'])
    plt.plot(history['test_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy_aami.png', dpi=300)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss_aami.png', dpi=300)
    plt.show()


def model_select(model_name, num_classes=5):
    # if model_name == 'LightX3ECG':
    #     model = LightX3ECG(num_classes=num_classes)
    # elif model_name == 'ecgTransForm':
    #     model = ecgTransForm()
    # elif model_name == 'MSDNN':
    #     model = MSDNN(num_classes, 1)
    # elif model_name == 'CMM':
    #     model = MyNet().to(device)
    # else:
    width_multiplier = 0.5
    resolution_multiplier = 0.5
    model = Model(width_multiplier, resolution_multiplier)

    return model

def compute_fold_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    per_class_f1 = f1_score(
        y_true, y_pred,
        average=None,
        labels=[0, 1, 2, 3, 4],
        zero_division=0
    )

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1N": per_class_f1[0],
        "F1S": per_class_f1[1],
        "F1V": per_class_f1[2],
        "F1F": per_class_f1[3],
        "F1Q": per_class_f1[4],
        "Favg": np.mean(per_class_f1)
    }
    return metrics


def plot_heat_map_row_normalized(y_true, y_pred, fold_id, save_dir='cv_results'):
    import os
    os.makedirs(save_dir, exist_ok=True)

    con_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(AAMI_CLASSES))))
    con_mat = con_mat.astype(np.float64)

    row_sums = con_mat.sum(axis=1, keepdims=True)
    con_mat_norm = np.divide(con_mat, row_sums, out=np.zeros_like(con_mat), where=row_sums != 0)

    plt.figure(figsize=(8, 8))
    ax = seaborn.heatmap(
        con_mat_norm,
        annot=True,
        fmt='.4f',
        cmap='Blues',
        xticklabels=AAMI_CLASSES,
        yticklabels=AAMI_CLASSES,
        annot_kws={"size": 14}   # 矩阵里数字变大
    )

    ax.set_xlabel('Predicted labels', fontsize=16)   # x轴标题变大
    ax.set_ylabel('True labels', fontsize=16)        # y轴标题变大
    ax.set_title(f'Fold {fold_id} Row-normalized Confusion Matrix', fontsize=16)  # 标题变大

    ax.tick_params(axis='x', labelsize=16)  # x轴类别标签变大
    ax.tick_params(axis='y', labelsize=16)  # y轴类别标签变大

    plt.tight_layout()
    plt.savefig(f'{save_dir}/fold_{fold_id}_confusion_matrix_row_norm.png', dpi=300)
    plt.close()

    return con_mat_norm

def model_select(model_name):
    # if model_name == 'LightX3ECG':
    #     model = LightX3ECG(num_classes=5)
    # elif model_name == 'ecgTransForm':
    #     model = ecgTransForm()
    # elif model_name == 'MSDNN':
    #     model = MSDNN(5, 1)
    # elif model_name == 'CMM':
    #     model = MyNet().to(device)
    # else:
    width_multiplier = 0.5
    resolution_multiplier = 0.5
        # model = Model(width_multiplier, resolution_multiplier)
    model = Model(
        width_multiplier = width_multiplier,
        resolution_multiplier = resolution_multiplier
    )
    return model
