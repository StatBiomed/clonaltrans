import torch
from .utils import get_topo_obs
from .config import Configuration
from .clonaltrans import CloneTranModel

def main(config, device):
    t_observed = torch.tensor([3.0, 10.0, 17.0]).to(device)
    t_observed = (t_observed - t_observed[0]) / (t_observed[-1] - t_observed[0])

    paga, array_total = get_topo_obs(data_dir='./data', fill_diagonal=config.paga_diagonal, device=device)
    print (array_total.mean())

    model = CloneTranModel(N=array_total, L=paga, config=config).to(device=device)
    model.train_model(t_observed)

    model.writer.flush()
    model.writer.close()
    return model