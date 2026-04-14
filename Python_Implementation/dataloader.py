import torch
import wfdb
import pywt
import numpy as np
from collections import Counter

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


def load_all_data():
    numberSet = [
        '100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
        '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
        '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
        '231', '232', '233', '234'
    ]
    # numberSet = [
    # '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    # '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    # '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    # '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    # '222', '223', '228', '230', '231', '232', '233', '234'
    # ]

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
