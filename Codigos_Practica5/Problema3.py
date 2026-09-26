import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


T = 200
betas = torch.linspace(0.0001, 0.02, T, device=device)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)


def proceso_directo(x0, t, eps):
    ab = alphas_bar[t - 1].view(-1, 1, 1, 1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps


def perdida(modelo, x0):
    B = x0.shape[0]
    t = torch.randint(1, T + 1, (B,), device=x0.device)
    eps = torch.randn_like(x0)
    xt = proceso_directo(x0, t, eps)
    eps_pred = modelo(xt, t)
    return F.mse_loss(eps_pred, eps)
