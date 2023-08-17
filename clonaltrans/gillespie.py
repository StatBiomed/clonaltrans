import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.lines import Line2D
import seaborn as sns

def gseed(seed):
    random.seed(seed)
    np.random.seed(seed)

def ginput_rates():
    #* Define rates (day to the minus 1)
    # Proliferation
    p0, p1, p2, p3, p4 = 0.02, 0.12, 0.3, -0.05, -0.2

    # Differentiation 
    r0_1, r0_2, r1_3, r2_4 = 0.015, 0.005, 0.1, 0.2

    # Group parameters
    diff_rates = np.array([r0_1, r0_2, r1_3, r2_4])
    prol_rates = np.array([p0, p1, p2, p3, p4])
    # Possible reactions for the gillespie
    rates = np.concatenate((diff_rates, np.abs(prol_rates)))
    number_diff = 4

    # Differentiation matrix
    M = np.zeros((4, 2), dtype=int)
    M[0, :] = [0, 1] 
    M[1, :] = [0, 2]
    M[2, :] = [1, 3]
    M[3, :] = [2, 4]

    return M, prol_rates, rates, number_diff

def ginput_state(M, num_pops):
    #* Cluster names, cell type id
    cluster_idxs = np.arange(0, num_pops)
    #* Index of clusters corresponding to the reactions
    vec_clusters = np.concatenate((M[:, 0], cluster_idxs)) 
    return cluster_idxs, vec_clusters

def gmodule(rates, number_cells, vec_clusters, t):
    #* the total propensity score & probability of each reaction
    rates_vec = np.sum(rates * np.array([number_cells[i] for i in vec_clusters]))
    r = (rates * np.array([number_cells[i] for i in vec_clusters])) / rates_vec
    
    # randomly choose when next reaction will occur and calculate time increment
    extract_time = random.random()
    delta_t = -np.log(extract_time) / rates_vec 
    t += delta_t 

    # randomly choose which reaction will occur
    extract_reaction = random.random()
    aux = extract_reaction - np.cumsum(r)
    aux[aux >= 0] = -10

    idx_reaction = np.argmax(aux)
    return idx_reaction, t, delta_t

def gillespie_main(
    seed, 
    start: float = 0,
    end: float = 50.1,
    step: float = 0.1,
    num_init_cells: int = 1,
    show: bool = True
):
    gseed(seed)
    time_all = np.arange(start, end, step) 

    if os.path.exists(f'./gillespie/all_cells_{seed}.csv'):
        os.remove(f'./gillespie/all_cells_{seed}.csv')
    if os.path.exists(f'./gillespie/summary_{seed}.csv'):
        os.remove(f'./gillespie/summary_{seed}.csv')
    if os.path.exists(f'./gillespie/occurred_{seed}.csv'):
        os.remove(f'./gillespie/occurred_{seed}.csv')

    M, prol_rates, rates, number_diff = ginput_rates()
    cluster_idxs, vec_clusters = ginput_state(M, len(prol_rates))
    occurred_reactions = np.zeros(len(rates))

    number_cells = np.zeros(len(cluster_idxs))
    number_cells[0] = num_init_cells #* Start with one stem cell *#

    all_cells = np.ones(len(time_all)) * np.sum(number_cells)
    all_cells_pops = np.zeros((len(time_all), len(cluster_idxs)))
    all_cells_pops[:, 0] = number_cells[0]

    old_index = 0
    t = 0 

    summary = pd.DataFrame(columns=['Time', 'Increment t', 'Reaction ID', 'Index old', 'Index new'])

    list_number_cells = [list(number_cells)]
    while np.sum(number_cells) > 0 and t <= time_all[-1]:
        idx_reaction, t, delta_t = gmodule(rates, number_cells, vec_clusters, t)
        occurred_reactions[idx_reaction] += 1

        if idx_reaction >= number_diff:
            if prol_rates[idx_reaction - number_diff] > 0:
                number_cells[idx_reaction - number_diff] += 1
            else:
                number_cells[idx_reaction - number_diff] -= 1
            
        else:
            number_cells[M[idx_reaction, 0]] -= 1
            number_cells[M[idx_reaction, 1]] += 1

        list_number_cells.append(list(number_cells))
        old_index_temp = old_index

        if t <= time_all[-1]:
            new_index = np.argmin(np.abs(time_all - t))
            all_cells[new_index:] = np.sum(number_cells)
            all_cells_pops[new_index:, :] = np.tile(number_cells, (len(time_all) - new_index, 1))
            old_index = new_index + 1
            
        else:
            new_index = len(time_all)
            # all_cells[-1] = np.sum(number_cells)
            # all_cells_pops[-1, :] = number_cells

        row = {
            'Time': t, 
            'Increment t': delta_t,
            'Reaction ID': idx_reaction, 
            'Index old': old_index_temp, 
            'Index new': new_index
        }
        summary.loc[len(summary)] = row

    all_cells = np.concatenate([all_cells_pops, all_cells.reshape(-1, 1)], axis=1)
    cols = [f'Cluster {i}' for i in cluster_idxs]
    cols.append('Total cells')

    cell_time = pd.DataFrame(
        columns=cols,
        index=range(all_cells.shape[0]),
        data=all_cells
    )
    cell_time.to_csv(f'./gillespie/all_cells_{seed}.csv')
    summary.to_csv(f'./gillespie/summary_{seed}.csv')
    
    with open(f'./gillespie/occurred_{seed}.csv', 'w') as f:
        np.savetxt(f, occurred_reactions)

    if show:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle('Gillespie Summary')
        gplot_summary(axes, summary, np.stack(list_number_cells))
        plt.show()

def gplot_summary(axes, summary, list_number_cells):
    time = np.concatenate(([0.], summary['Time'].values))
    axes[0].plot(time, np.sum(list_number_cells, axis=1), 'o', color='lightcoral')
    axes[0].set_xlabel('Experimental time course ($t$)')
    axes[0].set_ylabel('Total # of cells')

    increment = np.concatenate(([0.], summary['Increment t'].values))
    axes[1].plot(time, increment, 'x', color='#2C6975')
    axes[1].set_xlabel('Experimental time course ($t$)')
    axes[1].set_ylabel(f'Increment $ \Delta t$')

    for cell_id in range(list_number_cells.shape[1]):
        axes[2].plot(time, list_number_cells[:, cell_id], label=f'Cell {cell_id}')
    axes[2].set_xlabel('Experimental time course ($t$)')
    axes[2].set_ylabel('# of cells per population')
    plt.legend()

def generation_proliferation(pop1=0, pop2=0):
    kk_pop1, kk_pop2 = 0, 0
    wt_pop1, wt_pop2 = [], []

    for csvs in os.listdir('./gillespie'):
        if csvs.startswith('all_cells_'):
            all_cells = pd.read_csv(os.path.join('./gillespie', csvs), index_col=0)

            if np.sum(all_cells.values[:, pop1]) > 0:
                kk_pop1 += 1
                wt_pop1.append(np.where(all_cells.values[:, pop1] > 0)[0][0])

            if np.sum(all_cells.values[:, pop2]) > 0:
                kk_pop2 += 1 
                wt_pop2.append(np.where(all_cells.values[:, pop2] > 0)[0][0])
    
    print (f'Mean1: {np.mean(wt_pop1)}', f'Mean2: {np.mean(wt_pop2)}')
    print (f'Length1: {len(wt_pop1)}', f'Length2: {len(wt_pop2)}') 

def gillespie_analysis():
    time_all = np.arange(0, 50.1, 0.1)
    number_diff = 4

    across_seeds = []
    across_occur = []

    for csvs in os.listdir('./gillespie'):
        if csvs.startswith('all_cells_'):
            all_cells = pd.read_csv(os.path.join('./gillespie', csvs), index_col=0)
            across_seeds.append(all_cells.values)

        if csvs.startswith('occurred_'):
            summary = pd.read_csv(os.path.join('./gillespie', csvs), header=None)
            across_occur.append(list(summary.values.squeeze()))
        
    across_seeds = np.stack(across_seeds) #* (seeds, time points, clusters)
    across_occur = np.stack(across_occur) #* (seeds, num_reactions)

    mean_all = np.mean(across_seeds[:, :, -1], axis=0)
    mean_pop = np.mean(across_seeds[:, :, :-1], axis=0)

    fig, axes = plt.subplots(1, 1)
    axes.plot(time_all, across_seeds[:, :, -1].T, '#2C6975', linewidth=1) 
    axes.plot(time_all, mean_all, 'lightcoral', linewidth=2)
    axes.set_yscale('log')
    axes.set_xlabel('Experimental time')
    axes.set_ylabel('# cells')
    axes.set_xlim([time_all[0], time_all[-1]])
    legend_elements = [Line2D([0], [0], color='#2C6975', lw=1), Line2D([0], [0], color='lightcoral', lw=2)]
    labels = ['Mean total cells per seed', 'Mean total cells of all seeds']
    plt.legend(legend_elements, labels)

    fig, axes = plt.subplots(1, 1)
    sns.histplot(across_seeds[:, :, -1][:, -1], bins=30)
    axes.set_title(f'Mean cells of all seeds at $t$[-1] ' + str(np.mean(across_seeds[:, :, -1][:, -1])))
    axes.set_xlabel(f'Mean cells per seed at $t$[-1]')

    num_clusters = across_seeds[:, :, :-1].shape[2]
    fig, axes = plt.subplots(1, num_clusters, figsize=(24, 6))
    for cid in range(num_clusters):
        mm = np.mean(across_seeds[:, :, :-1][:, -1, cid])
        sns.histplot(across_seeds[:, :, :-1][:, -1, cid], ax=axes[cid])
        axes[cid].set_title('Cluster ' + str(cid) + ': ' + str(mm))
        axes[cid].set_xlabel(f'Mean cells per seed at $t$[-1]')

    colors = ['black','r','b','y','g']
    mat_clones = np.sum(across_seeds[:, :, :-1] > 0, axis=0).squeeze()

    fig, axes = plt.subplots(1, num_clusters, figsize=(24, 6))
    for cid in range(num_clusters):
        col = colors[cid]
        axes[cid].plot(time_all, mat_clones[:, cid], color=col, linewidth=2, label=cid)
        axes[cid].set_xlim(0, time_all[-1]) 
        axes[cid].set_ylim(np.min(mat_clones[:, cid]), np.max(mat_clones[:, cid]))
        axes[cid].set_title('Cluster ' + str(cid))
        axes[cid].set_ylabel('# different clones')
        axes[cid].set_xlabel('Experimental time')

    mat_clones2 = np.sum(across_seeds[:, :, -1] > 0, axis=0)

    fig, axes = plt.subplots(1, 1) 
    axes.plot(time_all, mat_clones2, color='#2C6975', linewidth=2)
    axes.set_xlim(0, time_all[-1]) 
    axes.set_ylim(np.min(mat_clones2), np.max(mat_clones2)) 
    axes.set_title('All clusters combined')
    axes.set_ylabel('# different clones')
    axes.set_xlabel('Experimental time')

    fig, axes = plt.subplots(1, num_clusters, figsize=(24, 6))
    for cid in range(num_clusters):
        col = colors[cid]
        axes[cid].plot(time_all, mean_pop[:, cid], col)
        axes[cid].set_xlim(0, time_all[-1]) 
        axes[cid].set_ylim(np.min(mean_pop[:, cid]), np.max(mean_pop[:, cid]))
        axes[cid].set_title('Cluster ' + str(cid))
        axes[cid].set_ylabel('Mean cells across seeds')
        axes[cid].set_xlabel('Experimental time')

    store_diff = np.sum(across_occur[:, :number_diff], axis=1).squeeze() #* (seeds, )
    store_prol = np.sum(across_occur[:, number_diff:], axis=1).squeeze()
    store_pops = np.sum(across_occur[:, :number_diff])

    # fig, axes = plt.subplots(1, num_clusters, figsize=(24, 6))
    # for cid in range(num_clusters):
    #     col = colors[cid]
    #     sns.histplot(store_pops[:, cid], color=col, ax=axes[cid])
    #     axes[cid].set_title('Cluster ' + str(cid))
    #     axes[cid].set_ylabel('Mean cells across seeds')
    #     axes[cid].set_xlabel('# of occured times')

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    sns.histplot(store_diff, ax=axes[0], bins=30)
    axes[0].set_xlabel('total # of differentiations')
    axes[0].set_ylabel('# of seed trails')
    sns.histplot(store_prol, ax=axes[1], bins=30)
    axes[1].set_xlabel('total # of proliferations')
    axes[1].set_ylabel('# of seed trails')

    fig, axes = plt.subplots(3, 3, figsize=(14, 18))
    for j in range(across_occur.shape[1]):
        sns.histplot(across_occur[:, j], ax=axes[j // 3, j % 3])
        axes[j // 3, j % 3].set_yscale('log')
        axes[j // 3, j % 3].set_title('Reaction ID: ' + str(j))
        axes[j // 3, j % 3].set_ylabel('# of seed trails')
        axes[j // 3, j % 3].set_xlabel('# of occurred times')

def temp2():
    import numpy as np

    # Sample data 
    tree = np.loadtxt('./trees/store_134.txt')

    # Preallocate matrices
    M = np.zeros((9,2), dtype=int)
    c = np.zeros((tree.shape[1] - 4, tree.shape[0]))  
    p = np.zeros(tree.shape[1] - 3)

    # Populate M
    M[0,:] = [1, 2]
    M[1,:] = [1, 3] 
    M[2,:] = [2, 4]
    M[3,:] = [3, 5]
    M[4,:] = [1, 1]
    M[5,:] = [2, 2]
    M[6,:] = [3, 3]
    M[7,:] = [4, 4]
    M[8,:] = [5, 5]

    # Loop through tree
    for sco in range(2, tree.shape[1] - 3):
    
            cell = sco
    temp = sco
    
    while cell > 1:
    
        min_temp = np.where(tree[0::2, 2+cell])[0][0] + 1
        r = tree[min_temp, 1]
        
        if r >= 5:
            p[sco] += 1
        
        cell = tree[min_temp, 2]

    import matplotlib.pyplot as plt
    import numpy as np

    # Colors 
    colors = ['0.5','r','b','y','g']

    # Plot empty circle 
    fig, ax = plt.subplots(1, 1, num=1)
    ax.plot(0,0,'o', color=[0.5,0.5,0.5], markerfacecolor=[0.5,0.5,0.5], markersize=20)

    # Initialize arrays
    centres = []
    times = []
    offset = 2**(max(p)-1)

    for j in range(0, tree.shape[0], 2):

        kk = tree[j, 2]
    col = colors[int(M[int(tree[j, 1]), 1])-1]
    
    if (tree[j, 1] >= 5) and (tree[j, 1] <= 7):
        
        centres.extend([centres[kk]-offset, centres[kk]+offset]) 
        times.extend([tree[j, 0], tree[j, 0]])

    # Plotting code goes here...

    for j in range(0, tree.shape[0], 2):

        kk = tree[j, 2]
    col = colors[int(M[int(tree[j, 1]), 1])-1]

    if (tree[j, 1] >= 5) and (tree[j, 1] <= 7):
        
        if j==0:
            ax.plot([centres[kk], centres[-1]], [0, tree[j,0]], color='k')
            ax.plot([centres[kk], centres[-2]], [0, tree[j,0]], color='k')
        else:
            ax.plot([centres[kk], centres[-1]], [times[kk], tree[j,0]], color='k')
            ax.plot([centres[kk], centres[-2]], [times[kk], tree[j,0]], color='k')

        offset = offset/2

        ax.plot(centres[-2], tree[j,0], 'o', color=col, markerfacecolor=col, markersize=20)
        ax.plot(centres[-1], tree[j,0], 'o', color=col, markerfacecolor=col, markersize=20)

    elif (tree[j,1] >= 8):

        if j==0: 
            ax.plot([centres[kk], centres[kk]], [0, tree[j,0]], color='k')
        else:
            ax.plot([centres[kk], centres[kk]], [times[kk], tree[j,0]], color='k')
    
            ax.plot(centres[kk], tree[j,0], 'x', color='k', markerfacecolor='k', markersize=20)

    else:
        centres.append(centres[kk])
    times.append(tree[j,0])

    for j in range(0, tree.shape[0], 2):

        if j==0:
            ax.plot([centres[kk], centres[-1]], [0, tree[j,0]], color='k')
        else:
            ax.plot([centres[kk], centres[-1]], [times[kk], tree[j,0]], color='k')
    
    ax.plot(centres[-1], tree[j,0], 'o', color=col, markerfacecolor=col, markersize=20)

    # Horizontal lines  
    # qq = min(centres)
    # for j2 in range(0, tree.shape[0], 2):
    #   ax.plot([qq-1, -qq+1], [tree[j2,0], tree[j2,0]], color='k', linestyle='--')

    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_ylim([-5, tree[-1,0] + 5])
    ax.set_ybound(upper=0)

    if min(centres) == 0:
        qq = 0.1
    else:
        qq = 1/max(np.abs(centres))*3

    # Set x limits
    ax.set_xlim([min(centres)-qq, max(centres)+qq])

    # Figure export
    fig.set_size_inches(8.5, 11)
    fig.savefig('./figures_paper/tree_{}.eps'.format(tt), 
                format='eps',
                dpi=1200,
                bbox_inches='tight')

    plt.close(fig)

def temp():
    kk = 0
    for sc in index_first:
    
        store_index = [sc+1, sc]
    i = sc-1
    
    while i>1:
    
        cell = vec_tree[i+1,1,3]  
        index = np.where(vec_tree[1:i-1,1,cell+3] == 0)[-1][-1] + 1
    
        store_index.append(index)
        
        i = index
            
    temp = vec_tree[store_index[2:]+1,1,2]
    
    temp2 = np.zeros(9)
    for jk in range(9):
        temp2[jk] = np.sum(temp == jk) 
    
    store_rates_first[kk,:] = [vec_unique[kk], temp2]
    
    store_prol_first[kk,:] = [vec_unique[kk], np.sum(temp >= 5)]
    store_dif_first[kk,:] = [vec_unique[kk], np.sum(temp <= 4)]
    
    kk += 1
    
    for pop in range(1,6):

        store_rates_first_mean[pop,:] = [seed, np.mean(store_rates_first[store_rates_first[:,0] == pop,1:], axis=0)]
    store_prol_first_mean[pop,:] = [seed, np.mean(store_prol_first[store_prol_first[:,0] == pop,1:], axis=0)]  
    store_dif_first_mean[pop,:] = [seed, np.mean(store_dif_first[store_dif_first[:,0] == pop,1:], axis=0)]