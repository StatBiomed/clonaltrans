import torch
from .utils import get_topo_obs
from .clonaltrans import CloneTranModel
from torch.utils.tensorboard import SummaryWriter

def run_model(
    config, 
    t_observed,
    trail_name: str = ''
):
    from .utils import set_seed, init_config_summary
    config = init_config_summary(config)
    set_seed(config.seed)

    writer = SummaryWriter(log_dir=None, comment=trail_name)

    paga, array_total = get_topo_obs(
        data_dir='./data', 
        fill_diagonal=config.paga_diagonal, 
        init_day_zero=config.init_day_zero,
        device=config.gpu
    )
    print (f'Mean of input data: {array_total.mean()}')

    model = CloneTranModel(
        N=array_total, 
        L=paga, 
        config=config,
        writer=writer
    ).to(config.gpu)

    t_observed = (t_observed - t_observed[0]) / (t_observed[-1] - t_observed[0])
    model.train_model(t_observed.to(config.gpu))

    model.writer.flush()
    model.writer.close()
    return model

