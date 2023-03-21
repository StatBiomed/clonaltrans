import torch
from .utils import get_topo_obs
from .clonaltrans import CloneTranModel
from torch.utils.tensorboard import SummaryWriter

def run_model(config, trail_name: str = ''):
    from .utils import set_seed, init_config_summary
    config = init_config_summary(config)
    set_seed(config.seed)

    writer = SummaryWriter(log_dir=None, comment=trail_name)

    t_observed = torch.tensor([3.0, 10.0, 17.0]).to(config.gpu)
    t_observed = (t_observed - t_observed[0]) / (t_observed[-1] - t_observed[0])

    paga, array_total = get_topo_obs(data_dir='./data', fill_diagonal=config.paga_diagonal, device=config.gpu)
    print (array_total.mean())

    model = CloneTranModel(
        N=array_total, 
        L=paga, 
        config=config,
        writer=writer
    ).to(config.gpu)
    model.train_model(t_observed)

    model.writer.flush()
    model.writer.close()
    return model

