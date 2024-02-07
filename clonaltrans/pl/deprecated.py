def diff_K_between_clones(model, K_type='const', index_pop=0, tpoint=1.0, direction='outbound'):
    K = model.get_matrix_K(K_type=K_type, eval=True, tpoint=tpoint).detach().cpu().numpy()
    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))
    pop_name = anno['populations'].values[index_pop]

    if direction == 'outbound':
        df = transit_K(model, K[:, index_pop, :], anno['clones'].values[:K.shape[0]], pop_name + ' -> ' + anno['populations'].values).T
    if direction == 'inbound':
        df = transit_K(model, K[:, :, index_pop], anno['clones'].values[:K.shape[0]], anno['populations'].values + ' -> ' + pop_name).T

    fig, axes = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, annot=True, linewidths=.5, cmap='coolwarm', ax=axes, vmin=-2, vmax=3)
    plt.xticks(rotation=0)
    plt.title(f'Difference in transition rates between clones for {pop_name} ({direction})')

def clone_dynamic_K(model, K_type='const', index_clone=0, suffix=''): 
    from PIL import Image
    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item() + 1)).to(model.config.gpu)
    
    if index_clone == -1: title = 'BG'
    else: title = index_clone

    frames = []
    for i in range(len(x)):
        clone_specific_K(model, K_type, index_clone, x[i].round(decimals=1), save=True)
        frames.append(Image.open(f'./figs/temp_{x[i].round(decimals=1)}.png'))
        os.remove(f'./figs/temp_{x[i].round(decimals=1)}.png')

    imageio.mimsave(f'K_dynamics_clone_{title}{suffix}.gif', frames, duration=500, loop=0)

def clone_dynamic_K_lines(model, index_clone=0, save=False):
    K_total = []
    N_total = []
    t_obs = model.t_observed.cpu().numpy()
    x = torch.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item()) + 1).to(0)

    model.t_observed = model.t_observed.to(0)
    
    for i in range(len(x)):
        K = model.get_matrix_K(K_type=model.config.K_type, eval=True, tpoint=x[i]).detach().cpu().numpy()
        K_total.append(K)

        N = model.eval_model(torch.Tensor([0.0, x[i] + 0.01 if x[i] == 0 else x[i]]))[1, index_clone]
        N = torch.pow(N, 1 / model.config.exponent)
        
        N_total.append(N)

    K_total = np.stack(K_total)
    N_total = np.stack(N_total)

    anno = pd.read_csv(os.path.join(model.data_dir, 'annotations.csv'))
    graph = pd.read_csv(os.path.join(model.data_dir, 'graph_table.csv'), index_col=0)
    np.fill_diagonal(graph.values, 1)

    rows, cols, figsize = get_subplot_dimensions(np.sum(graph.values != 0), fig_height_per_row=4)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] + 5, figsize[1]))

    count = 0
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph.values[i][j] != 0:   
                df = K_total[:, index_clone, i, j]             
                df[np.abs(df) < 1e-4] = 0
                df[np.where(N_total[:, i] < 0.5)[0]] = 0

                f = interp1d(np.linspace(0, int(model.t_observed[-1].item()), int(model.t_observed[-1].item()) + 1), df)
                newx = np.linspace(0, int(model.t_observed[-1].item()), 50)
                newy = f(newx)

                axes[count // cols][count % cols].plot(
                    newx, 
                    newy, 
                    color='#2C6975',
                    lw=4,
                )

                axes[count // cols][0].set_ylabel('Per capita transition rate', fontsize=13)
                axes[count // cols][count % cols].set_title('From {} to {}'.format(anno['populations'].values[i], anno['populations'].values[j]), fontsize=13)
                axes[count // cols][count % cols].set_xticks(t_obs, labels=t_obs.astype(int))
                count += 1
    
    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')