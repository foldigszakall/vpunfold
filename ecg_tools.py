import torch
import numpy as np
import scipy.io
import random
from typing import  Callable, Optional
import itertools
import sys
import os

class Logger:
    original_stdout = sys.stdout
    def __init__(self, *streams):
        self.streams = [self.original_stdout, *streams]
    def __enter__(self):
        sys.stdout = self
        return self
    def __exit__(self, *args):
        sys.stdout = self.original_stdout
    def write(self, message):
        for f in self.streams:
            f.write(message)

def confusion_matrix(y_true, y_pred, n_classes):
    w = torch.ones_like(y_true)
    y = torch.stack((y_true, y_pred))
    return torch.sparse_coo_tensor(y, w, (n_classes, n_classes)).to_dense()

def ecg_dataset(path: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) \
                -> torch.utils.data.Dataset:
    mat = scipy.io.loadmat(path)
    samples = torch.tensor(mat['samples'], device=device, dtype=dtype)
    samples = samples.unsqueeze(1)
    if 'rr' in mat:
        rr = torch.tensor(mat['rr'], device=device, dtype=dtype)
        rr = rr.unsqueeze(1) / 360
    else:
        rr = torch.zeros((samples.size(0), 1, 0), device=device, dtype=dtype)
    labels = torch.tensor(mat['labels'], device=device, dtype=dtype)
    return torch.utils.data.TensorDataset(samples, rr, labels)

def train(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, \
          n_epoch: int, optimizer: torch.optim.Optimizer, \
          criterion: Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor]) -> None:
    n_digits = len(str(n_epoch))
    for epoch in range(n_epoch):
        total_loss = 0
        total_accuracy = 0
        total_number = 0
        for data in data_loader:
            x, rr, labels = data
            optimizer.zero_grad()
            outputs = model(x, rr)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            classes = labels.argmax(dim=-1)
            y = outputs[0] if isinstance(outputs, tuple) else outputs
            y_classes = y.argmax(dim=-1)
            total_accuracy += (classes == y_classes).sum().item()
            total_number += labels.size(0)
        total_accuracy /= total_number / 100
        print(f'Epoch: {epoch+1:0{n_digits}d} / {n_epoch}, accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}')

def test(target: str, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, \
         criterion: Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor]) -> float:
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        total_number = 0
        total_cm = 0
        for data in data_loader:
            x, rr, labels = data
            outputs = model(x, rr)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            classes = labels.argmax(dim=-1)
            y = outputs[0] if isinstance(outputs, tuple) else outputs
            y_classes = y.argmax(dim=-1)
            total_accuracy += (classes == y_classes).sum().item()
            total_number += labels.size(0)
            total_cm += confusion_matrix(classes, y_classes, labels.size(-1))
        total_accuracy /= total_number / 100
        print(f'{target} accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}')
        print(total_cm)
        return total_accuracy

def random_init(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def evaluate_models(model_builder: Callable[..., tuple[str, torch.nn.Module, torch.nn.Module]], \
                    train_set: torch.utils.data.Dataset, test_set: torch.utils.data.Dataset, \
                    batch_size: list[int], lr: list[float], epoch: list[int], **kwargs) -> tuple[float, str]:
    os.makedirs('models', exist_ok=True)
    best = 0.
    best_name = ''
    keys = kwargs.keys()
    for batch_size, lr, epoch in itertools.product(batch_size, lr, epoch):
        for item in itertools.product(*kwargs.values()):
            random_init()
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
            model_args = dict(zip(keys, item))
            name, model, criterion = model_builder(batch_size, lr, epoch, **model_args)
            accuracy = 0.
            with open(f'models/{name}.log', 'w') as f:
                with Logger(f):
                    print(f'{batch_size=}\n{lr=}\n{epoch=}\n')
                    print(model)
                    print(criterion)
                    try:
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                        train(model, train_loader, epoch, optimizer, criterion)
                        test('Train', model, train_loader, criterion)
                        accuracy = test('Test', model, test_loader, criterion)
                        torch.save(model.state_dict(), f'models/{name}.pt')
                    except Exception as err:
                        print(f'{type(err)}: {err}')
                    print()
            if accuracy > best:
                best = accuracy
                best_name = name
    return best, best_name