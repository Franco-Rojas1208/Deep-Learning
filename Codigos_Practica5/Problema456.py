import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando:", device)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        mitad = self.dim // 2
        frecuencias = torch.exp(-math.log(10000) * torch.arange(mitad, device=t.device) / mitad)
        args = t.float()[:, None] * frecuencias[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


class Bloque(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h


class UNet(nn.Module):
    def __init__(self, canales=1, base=32, t_dim=128):
        super().__init__()
        self.time_emb = TimeEmbedding(t_dim)

        self.entrada = nn.Conv2d(canales, base, 3, padding=1)

        self.down1 = Bloque(base, base, t_dim)
        self.down2 = Bloque(base, 2 * base, t_dim)
        self.pool = nn.MaxPool2d(2)

        self.medio = Bloque(2 * base, 4 * base, t_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up2 = Bloque(4 * base + 2 * base, 2 * base, t_dim)
        self.up1 = Bloque(2 * base + base, base, t_dim)

        self.salida = nn.Conv2d(base, canales, 1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        x = self.entrada(x)
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        m = self.medio(self.pool(d2), t_emb)

        u2 = self.upsample(m)
        u2 = self.up2(torch.cat([u2, d2], dim=1), t_emb)
        u1 = self.upsample(u2)
        u1 = self.up1(torch.cat([u1, d1], dim=1), t_emb)

        return self.salida(u1)


transformacion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_data = datasets.MNIST("data", train=True, download=True, transform=transformacion)
test_data = datasets.MNIST("data", train=False, download=True, transform=transformacion)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

modelo = UNet().to(device)
optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
epocas = 10

historial = []
for epoca in range(epocas):
    modelo.train()
    suma, n = 0.0, 0
    for x0, _ in train_loader:
        x0 = x0.to(device)
        loss = perdida(modelo, x0)

        optimizador.zero_grad()
        loss.backward()
        optimizador.step()

        historial.append(loss.item())
        suma += loss.item()
        n += 1
    print(f"Época {epoca + 1}/{epocas} - pérdida media: {suma / n:.4f}")

plt.figure(figsize=(6, 3.5))
plt.plot(historial, alpha=0.4, label="por batch")
ventana = 100
suavizada = [sum(historial[max(0, i - ventana):i + 1]) / len(historial[max(0, i - ventana):i + 1])
             for i in range(len(historial))]
plt.plot(suavizada, label=f"Suavización")
plt.xlabel("iteración"); plt.ylabel("pérdida"); plt.yscale("log")
plt.legend(); plt.grid(alpha=0.3); plt.title("Entrenamiento")
plt.show()


def estimar_x0(modelo, xt, t):
    ab = alphas_bar[t - 1].view(-1, 1, 1, 1)
    eps_pred = modelo(xt, t)
    return (xt - torch.sqrt(1 - ab) * eps_pred) / torch.sqrt(ab)

modelo.eval()
tiempos_eval = [10, 50, 100, 150, 200]
print("\n  t   pérdida (ruido)   MSE ruidosa   MSE reconstruida")
torch.manual_seed(0)
with torch.no_grad():
    for t_val in tiempos_eval:
        perd, mse_ruidosa, mse_rec, n = 0.0, 0.0, 0.0, 0
        for x0, _ in test_loader:
            x0 = x0.to(device)
            t = torch.full((x0.shape[0],), t_val, device=device)
            eps = torch.randn_like(x0)
            xt = proceso_directo(x0, t, eps)
            eps_pred = modelo(xt, t)
            x0_est = estimar_x0(modelo, xt, t).clamp(-1, 1)

            perd += ((eps_pred - eps) ** 2).mean().item()
            mse_ruidosa += ((xt.clamp(-1, 1) - x0) ** 2).mean().item()
            mse_rec += ((x0_est - x0) ** 2).mean().item()
            n += 1
        print(f"{t_val:3d}   {perd / n:15.4f}   {mse_ruidosa / n:11.4f}   {mse_rec / n:16.4f}")


def proceso_inverso(modelo, xt, t_inicial):
    x = xt
    for t in range(t_inicial, 0, -1):
        tt = torch.full((x.shape[0],), t, device=device)
        eps_pred = modelo(x, tt)
        media = (x - betas[t - 1] / torch.sqrt(1 - alphas_bar[t - 1]) * eps_pred) / torch.sqrt(alphas[t - 1])
        if t > 1:
            x = media + torch.sqrt(betas[t - 1]) * torch.randn_like(x)
        else:
            x = media
    return x.clamp(-1, 1)


def mostrar(x):
    return ((x + 1) / 2).squeeze().cpu()

x0_ejemplos = torch.stack([test_data[i][0] for i in range(6)]).to(device)
filas = ["original", "con ruido $x_t$", "un paso ($\\hat x_0$)", "proceso inverso"]

with torch.no_grad():
    for t_val in [50, 100, 200]:
        t = torch.full((6,), t_val, device=device)
        xt = proceso_directo(x0_ejemplos, t, torch.randn_like(x0_ejemplos))
        un_paso = estimar_x0(modelo, xt, t).clamp(-1, 1)
        inverso = proceso_inverso(modelo, xt, t_val)

        fig, axes = plt.subplots(4, 6, figsize=(9, 6.5))
        for j in range(6):
            for i, img in enumerate([x0_ejemplos, xt.clamp(-1, 1), un_paso, inverso]):
                axes[i, j].imshow(mostrar(img[j]), cmap="gray", vmin=0, vmax=1)
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                if j == 0:
                    axes[i, j].set_ylabel(filas[i], fontsize=9)
        fig.suptitle(f"t = {t_val}   (ᾱ = {alphas_bar[t_val - 1].item():.3f})")
        plt.tight_layout()
        plt.show()
