import torch 
from utils import set_seed

from trainer import CloneTranModel
from model import ODESolver

def run_model(
    config, 
    N, 
    L, 
    t_simu, 
    save_name='model_last.pt'
):
    set_seed(config['system']['seed'])
    N, L = N.to(config['system']['gpu_id']), L.to(config['system']['gpu_id'])

    model = ODESolver(
        L=L,
        num_clones=N.shape[1],
        num_pops=N.shape[2],
        hidden_dim=config['arch']['args']['hidden_dim'], 
        activation=config['arch']['args']['activation'], 
        K_type=config['arch']['args']['K_type'],
        adjoint=config['user_trainer']['adjoint']
    ).to(config['system']['gpu_id'])

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['optimizer']['learning_rate'], 
        weight_decay=config['optimizer']['weight_decay'],
        amsgrad=True
    )

    trainer = CloneTranModel(
        N=N, 
        L=L, 
        config=config,
        model=model,
        optimizer=optimizer,
        t_observed=t_simu.to(config['system']['gpu_id'], dtype=torch.float32),
        trainer_type='simulation',
        writer=None,
        sample_N=None,
        gpu_id=config['system']['gpu_id']
    )
    trainer.train_model()

    torch.save(trainer, save_name)