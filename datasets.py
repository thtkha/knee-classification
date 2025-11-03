from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import torch


def read_xray(path):
    """Read a grayscale xray, normalize and return a 3-channel (C,H,W) float32 numpy array.

    This function normalizes backslashes to the OS path separator, tries an absolute
    path fallback, and raises a FileNotFoundError with the resolved path if reading fails.
    """
    # Normalize the path string and replace backslashes (Windows-style) with OS separator
    path = str(path).strip()
    path = path.replace('\\', os.sep)
    path = os.path.normpath(path)

    # If the path is relative and doesn't exist, try resolving relative to cwd
    if not os.path.exists(path):
        alt = os.path.join(os.getcwd(), path.lstrip('./'))
        if os.path.exists(alt):
            path = alt

    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if xray is None:
        # Provide a helpful error so the caller knows which file failed
        raise FileNotFoundError(f"Failed to read image at path: {path}")

    xray = xray.astype(np.float32) / 255.0

    # Stack into 3 channels (C, H, W)
    xray_3ch = np.stack([xray, xray, xray], axis=0).astype(np.float32)

    return xray_3ch


class knee_xray_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = read_xray(self.dataset['Path'].iloc[index])
        label = int(self.dataset['KL'].iloc[index])

        # convert to torch tensors with correct dtypes
        img = torch.from_numpy(img)        # float32 tensor shaped (C, H, W)
        label = torch.tensor(label, dtype=torch.long)

        res = {
            'img': img,
            'label': label
        }
        return res
