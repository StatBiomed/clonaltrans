def run_model(
    config, 
    t_observed,
    trail_name: str = '',
    N: any = None,
    L: any = None
):
    from .utils import set_seed, init_config_summary
    from .datasets import get_topo_obs
    config = init_config_summary(config)
    set_seed(config.seed)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=None, comment=trail_name)

    paga, array_total = get_topo_obs(
        data_dir='./data', 
        init_day_zero=True,
        device=config.gpu
    )

    from .clonaltrans import CloneTranModel
    model = CloneTranModel(
        N=array_total if N == None else N, 
        L=paga if L == None else L, 
        config=config,
        writer=writer
    ).to(config.gpu)

    t_observed = (t_observed - t_observed[0]) / (t_observed[-1] - t_observed[0])
    model.train_model(t_observed.to(config.gpu))

    model.writer.flush()
    model.writer.close()
    return model

