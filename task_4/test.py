import torch

obs = torch.randn(3)
print(obs)
obs = obs.view(1, -1).expand(3, 3).t()
print(obs)
print(obs.sum(dim=0))
print(torch.logsumexp(obs, dim=0))
