def run_model(
    config, 
    t_observed,
    t_eval: any = None,
    trail_name: str = ''
):
    from .utils import set_seed, init_config_summary, get_topo_obs
    config = init_config_summary(config)
    set_seed(config.seed)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=None, comment=trail_name)

    paga, array_total = get_topo_obs(
        data_dir='./data', 
        fill_diagonal=True, 
        simulation=config.simulation,
        init_day_zero=True,
        device=config.gpu
    )

    from .clonaltrans import CloneTranModel
    model = CloneTranModel(
        N=array_total, 
        L=paga, 
        config=config,
        writer=writer
    ).to(config.gpu)

    if t_eval != None:
        t_eval = (t_eval - t_eval[0]) / (t_eval[-1] - t_eval[0])

    t_observed = (t_observed - t_observed[0]) / (t_observed[-1] - t_observed[0])
    model.train_model(t_observed.to(config.gpu), t_eval)

    model.writer.flush()
    model.writer.close()
    return model

