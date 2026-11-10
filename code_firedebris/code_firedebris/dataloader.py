import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # 모델 입력 규격에 맞게 [Samples, 1, Length]로 변환
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def apply_binning(data, bin_size=50):
    n = len(data)
    new_length = n // bin_size
    reshaped_data = data[:new_length * bin_size].reshape(new_length, bin_size)
    return np.max(reshaped_data, axis=1)

def load_and_preprocess_data(data_dir, max_length, bining=True, log_transform=False, normalization="Zscore", encoder=None):

    if encoder is not None:
        category_list = encoder.classes_.tolist()
    else:
        category_list = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    file_info = []
    for cat in category_list:
        path = os.path.join(data_dir, cat)
        if not os.path.exists(path):
            continue
        target_files = sorted(glob.glob(os.path.join(path, '*.csv')) + glob.glob(os.path.join(path, '*.xlsx')))
        for f in target_files:
            file_info.append((f, cat))

    num_samples = len(file_info)

    # 🔵 핵심: shape 먼저 결정
    if bining:
        X = np.zeros((num_samples, max_length // 25), dtype=np.float32)
    else:
        X = np.zeros((num_samples, max_length), dtype=np.float32)

    labels = []
    file_paths = []

    for i, (f_path, cat_name) in enumerate(tqdm(file_info, desc=f"Loading data {os.path.basename(data_dir)}")):

        if f_path.endswith('.csv'):
            df = pd.read_csv(f_path, header=None)
        else:
            df = pd.read_excel(f_path, header=None)

        data = pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0).values

        if log_transform:
            data = np.log1p(data)

        temp = np.zeros(max_length, dtype=np.float32)
        length = min(len(data), max_length)
        temp[:length] = data[:length]
        data = temp

        # normalization
        if normalization == "Zscore":
            std = np.std(data)
            if std > 0:
                data = (data - np.mean(data)) / std
        elif normalization == "MinMax":
            div = np.max(data) - np.min(data)
            if div > 0:
                data = (data - np.min(data)) / div
        elif normalization == "MaxAbs":
            s_max = np.max(np.abs(data))
            if s_max > 0:
                data = data / s_max

        # 🔵 여기서 binning
        if bining:
            data = apply_binning(data, bin_size=25)

        X[i, :] = data
        # if log_transform:
        #      X[i, :] = np.log1p(X[i, :])

        labels.append(cat_name)
        file_paths.append(f_path)

    if encoder is None:
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
    else:
        y = encoder.transform(labels)

    return X, y, encoder, file_paths