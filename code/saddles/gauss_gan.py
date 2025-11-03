import torch
import torch.nn as nn
import torch.optim as optim
from optimizers import Muon, NormalizedMuon, Neon

# --- 1. Simple generator and discriminator ---
class Generator(nn.Module):
    def __init__(self, z_dim=2, hidden=16, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim=2, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- 2. Models and optimizers ---
z_dim = 2
G = Generator(z_dim)
D = Discriminator()
device = "cuda" if torch.cuda.is_available() else "cpu"
G, D = G.to(device), D.to(device)

#optim_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
#optim_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))
# optim_G = optim.SGD(G.parameters(), lr=1e-4, momentum=0.9)
# optim_D = optim.SGD(D.parameters(), lr=1e-4, momentum=0.9)
optim_G = Neon(G.parameters(), lr=0.05, momentum=0.9, sgd_coeff=1)
optim_D = Neon(D.parameters(), lr=0.05, momentum=0.9, sgd_coeff=1) # sgd_coeff = 0.5 or 1 -- 1.604 0.327; sgd_coeff = 0: 1500, 0.6
# optim_G = NormalizedMuon(G.parameters(), lr=0.05, momentum=0.9, sgd_coeff=0.5)
# optim_D = NormalizedMuon(D.parameters(), lr=0.05, momentum=0.9, sgd_coeff=0.5) -- 1.606 0.325
optim_G = NormalizedMuon(G.parameters(), lr=0.025, momentum=0.9, sgd_coeff=0)
optim_D = NormalizedMuon(D.parameters(), lr=0.025, momentum=0.9, sgd_coeff=0) # -- no limit cycles; about 1.3 and 0.9


# --- 3. Toy dataset: real data from a Gaussian ---
real_dist = torch.distributions.Normal(torch.tensor([2.0, 0.0]), torch.tensor([0.5, 0.5]))

# --- 4. Training loop ---
for step in range(10000):
    # Train discriminator
    real = real_dist.sample((64,)).to(device)
    z = torch.randn(64, z_dim).to(device)
    fake = G(z).detach()

    D_loss = -torch.mean(torch.log(D(real) + 1e-8) + torch.log(1 - D(fake) + 1e-8))
    optim_D.zero_grad()
    D_loss.backward()
    optim_D.step()

    # Train generator
    z = torch.randn(64, z_dim).to(device)
    fake = G(z)
    G_loss = -torch.mean(torch.log(D(fake) + 1e-8))
    optim_G.zero_grad()
    G_loss.backward()
    optim_G.step()

    if step % 1000 == 0:
        print(f"Step {step}: D_loss={D_loss.item():.3f}, G_loss={G_loss.item():.3f}")
