# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tqdm.auto import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 0

BATCH_SIZE = 128
N_CLASES = 10
LEARNING_RATE = 1e-3
EPOCHS = 10

torch.manual_seed(SEED)
np.random.seed(SEED)


transform = transforms.ToTensor()

train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

usar_gpu = DEVICE.type == "cuda"
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=usar_gpu)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                         num_workers=2, pin_memory=usar_gpu)

x0, y0 = train_ds[0]
print(f"\ntrain: {len(train_ds)} imagenes   test: {len(test_ds)} imagenes")
print(f"forma de una imagen: {tuple(x0.shape)} (C,H,W)")
print(f"rango de x: [{x0.min():.1f}, {x0.max():.1f}]   etiqueta del ejemplo: {int(y0)}")
print(f"batches por epoca: {len(train_loader)} (batch_size={BATCH_SIZE})")


class NN(nn.Module):

    def __init__(self, n_clases=N_CLASES):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_clases)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


model = NN(N_CLASES).to(DEVICE)
print("\n" + str(model))


criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n{criterion}")
print(optimizer)


def train_una_epoca(model, loader, criterion, optimizer, epoca=None):
    model.train()
    perdida_acum, correctos, total = 0.0, 0, 0

    barra = tqdm(loader, desc=f"epoca {epoca}", leave=False)
    for x, y in barra:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        perdida_acum += loss.item() * x.size(0)
        correctos += (out.argmax(dim=1) == y).sum().item()
        total += x.size(0)

        barra.set_postfix(loss=f"{perdida_acum/total:.4f}",
                          acc=f"{100*correctos/total:.2f}%")

    return perdida_acum / total, correctos / total


@torch.no_grad()
def evaluar(model, loader, criterion):
    model.eval()
    perdida_acum, correctos, total = 0.0, 0, 0
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        out = model(x)
        loss = criterion(out, y)

        pred = out.argmax(dim=1)
        perdida_acum += loss.item() * x.size(0)
        correctos += (pred == y).sum().item()
        total += x.size(0)

        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    return (perdida_acum / total,
            correctos / total,
            np.concatenate(y_true),
            np.concatenate(y_pred))


print(f"\nEntrenando: epocas={EPOCHS}, batch_size={BATCH_SIZE}, "
      f"lr={LEARNING_RATE}, device={DEVICE}\n")

hist = {"loss": [], "acc": [], "test_loss": [], "test_acc": []}

for epoca in range(1, EPOCHS + 1):
    loss_tr, acc_tr = train_una_epoca(model, train_loader, criterion, optimizer, epoca)
    loss_te, acc_te, _, _ = evaluar(model, test_loader, criterion)

    hist["loss"].append(loss_tr)
    hist["acc"].append(acc_tr)
    hist["test_loss"].append(loss_te)
    hist["test_acc"].append(acc_te)

    print(f"  epoca {epoca:2d}/{EPOCHS}  "
          f"loss={loss_tr:.4f}  acc={100*acc_tr:.2f}%   |   "
          f"test_loss={loss_te:.4f}  test_acc={100*acc_te:.2f}%")


epocas = np.arange(1, EPOCHS + 1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(epocas, hist["loss"], "o-", label="train")
ax[0].plot(epocas, hist["test_loss"], "o-", label="test")
ax[0].set_xlabel("epoca"); ax[0].set_ylabel("perdida (NLL)")
ax[0].set_title("Perdida"); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(epocas, 100 * np.array(hist["acc"]), "o-", label="train")
ax[1].plot(epocas, 100 * np.array(hist["test_acc"]), "o-", label="test")
ax[1].set_xlabel("epoca"); ax[1].set_ylabel("accuracy [%]")
ax[1].set_title("Precision"); ax[1].legend(); ax[1].grid(alpha=0.3)

fig.suptitle("Ejercicio 2 - CNN sobre MNIST (PyTorch)")
fig.tight_layout()
plt.show()


loss_test, acc_test, y_true, y_pred = evaluar(model, test_loader, criterion)

print(f"\nperdida test  = {loss_test:.4f}")
print(f"accuracy test = {acc_test:.4f}  ({100 * acc_test:.2f} %)")
print(f"errores: {int((y_true != y_pred).sum())} sobre {y_true.size}")


cm = confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASES))

fig, ax = plt.subplots(figsize=(6.5, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(N_CLASES))
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_xlabel("prediccion"); ax.set_ylabel("verdadero")
ax.set_title("Ejercicio 2 - matriz de confusion (test)")
fig.tight_layout()
plt.show()
