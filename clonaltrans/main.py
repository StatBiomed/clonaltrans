def run_model(
    config, 
    t_observed,
    t_norm: bool = False,
    data_dir: str = './data/V3_Mingze',
    N: any = None,
    L: any = None
):
    from .utils import set_seed, init_config_summary
    from .datasets import get_topo_obs
    config = init_config_summary(config)
    set_seed(config.seed)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=None)

    if N == None:
        paga, array_total = get_topo_obs(
            data_dir=data_dir, 
            init_day_zero=True,
            device=config.gpu
        )
    else:
        print (f'Shape of input: {N.shape}')
        paga, array_total = L, N

    from .clonaltrans import CloneTranModel
    model = CloneTranModel(
        N=array_total if N == None else N, 
        L=paga if L == None else L, 
        config=config,
        writer=writer
    ).to(config.gpu)

    if t_norm:
        t_observed = (t_observed - t_observed[0]) / (t_observed[-1] - t_observed[0]) * t_norm
        print (f'Original time scale of the system has been normalized with {t_norm}, {t_observed.cpu().numpy()}.')
    else:
        print (f'Integration time of ODE solver is {t_observed.cpu().numpy()}')

    model.t_observed = t_observed.to(config.gpu)
    model.train_model(model.t_observed)
    model.data_dir = data_dir

    model.writer.flush()
    model.writer.close()
    return model

