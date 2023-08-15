import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pandas as pd

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

def ginput_state(M):
    #* Cluster names, cell type id
    cluster_names = np.arange(0, 5)
    time_all = np.arange(0, 50.1, 0.1) 

    #* Index of clusters corresponding to the reactions
    vec_clusters = np.concatenate((M[:, 0], cluster_names)) 
    return cluster_names, time_all, vec_clusters

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

def gillespie_main(seed, show=False):
    gseed(seed)

    if os.path.exists(f'./gillespie/all_cells_{seed}.csv'):
        os.remove(f'./gillespie/all_cells_{seed}.csv')
    if os.path.exists(f'./gillespie/occurred_rates_{seed}.csv'):
        os.remove(f'./gillespie/occurred_rates_{seed}.csv')

    M, prol_rates, rates, number_diff = ginput_rates()
    cluster_names, time_all, vec_clusters = ginput_state(M)
    occurred_reactions = np.zeros(len(rates))

    all_cells = np.ones(len(time_all))
    all_cells_pops = np.zeros((len(time_all), len(cluster_names)))
    all_cells_pops[:, 0] = 1 

    old_index = 0
    t = 0 
    iters = 0
    number_cells = np.zeros(len(cluster_names))
    number_cells[0] = 1 # Start with one stem cell

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle('Gillespie Summary')

    summary = pd.DataFrame(columns=['Time', 'Increment t', 'Reaction ID', 'Outbound', 'Inbound', 'Index old', 'Index new'])

    list_t, list_delta_t, list_number_cells = [], [], []
    while np.sum(number_cells) > 0 and t <= time_all[-1]:
        idx_reaction, t, delta_t = gmodule(rates, number_cells, vec_clusters, t)
        occurred_reactions[idx_reaction] += 1

        if idx_reaction >= number_diff:
            outbound_cell = idx_reaction - number_diff
            inbound_cell = idx_reaction - number_diff

            if prol_rates[idx_reaction - number_diff] > 0:
                number_cells[idx_reaction - number_diff] += 1
            else:
                number_cells[idx_reaction - number_diff] -= 1
            
        else:
            outbound_cell = M[idx_reaction, 0]
            inbound_cell = M[idx_reaction, 1]
            number_cells[M[idx_reaction, 0]] -= 1
            number_cells[M[idx_reaction, 1]] += 1

        list_t.append(t)
        list_delta_t.append(delta_t)
        list_number_cells.append(list(number_cells))

        old_index_temp = old_index
        if t <= time_all[-1]:
            new_index = np.argmin(np.abs(time_all - t))
            all_cells[new_index:] = np.sum(number_cells)
            all_cells_pops[new_index:, :] = np.tile(number_cells, (len(time_all) - new_index, 1))
            old_index = new_index + 1
            
        else:
            new_index = len(time_all)
            all_cells[-1] = np.sum(number_cells)
            all_cells_pops[-1, :] = number_cells

        row = {
            'Time': t, 
            'Increment t': delta_t,
            'Reaction ID': idx_reaction, 
            'Outbound': outbound_cell, 
            'Inbound': inbound_cell, 
            'Index old': old_index_temp, 
            'Index new': new_index
        }
        summary.loc[len(summary)] = row
        iters += 1

    summary.to_csv(f'./gillespie/summary_{seed}.csv')
    all_cells = np.concatenate([all_cells.reshape(-1, 1), all_cells_pops], axis=1)
    with open(f'./gillespie/all_cells_{seed}.csv', 'a') as f1:
        np.savetxt(f1, all_cells, delimiter=',', fmt='%.2e')

    if show:
        gfig_config(axes, list_t, list_delta_t, np.stack(list_number_cells))
        plt.show()

def gfig_config(axes, t, delta_t, number_cells):
    axes[0].plot(t, np.sum(number_cells, axis=1), 'o', color='lightcoral')
    axes[0].set_xlabel('Experimental time course ($t$)')
    axes[0].set_ylabel('Total # of cells')

    axes[1].plot(t, delta_t, 'x', color='#2C6975')
    axes[1].set_xlabel('Experimental time course ($t$)')
    axes[1].set_ylabel(f'Increment $ \Delta t$')

    for cell_id in range(number_cells.shape[1]):
        axes[2].plot(t, number_cells[:, cell_id], label=f'Cell {cell_id}')
    axes[2].set_xlabel('Experimental time course ($t$)')
    axes[2].set_ylabel('# of cells per population')
    plt.legend()

def gvec_tree(vec_tree, t, idx_reaction, num_cell):
    vec_tree[1, :, :] = vec_tree[0, :, :]
    vec_tree[1, :, 0] = t
    vec_tree[1, :, 1] = idx_reaction
    vec_tree[1, :, 2] = num_cell
    vec_tree[1, 0, num_cell + 2] = 0
    return vec_tree

def gvec_pool(vec_tree, paras):
    pool = np.where(
        (np.squeeze(vec_tree[0, 0, 3:]) > 0) & 
        (np.squeeze(vec_tree[0, 1, 3:]) == paras)
    )
    num_cell = pool[np.random.randint(1, len(pool) + 1)]
    return num_cell

def gillespie_trees(seed):
    gseed(seed)
    M, prol_rates, rates, number_diff = ginput_rates()
    cluster_names, time_all, vec_clusters = ginput_state(M)

    jmax = 1

    #? meaning of j and for loop?
    for j in range(jmax):
        t = 0
        iters = 0
        number_cells = np.zeros(len(cluster_names))
        number_cells[0] = 1

        #? meaning of 1 2 4 and vec_tree?
        vec_tree = np.zeros((1, 2, 4))
        vec_tree[0, 0, :] = [0, 0, 1, 1]
        vec_tree[0, 1, :] = [0, 0, 1, 1]

        while np.sum(number_cells) > 0 and t <= time_all[-1]:
            idx_reaction, t, delta_t = gmodule(rates, number_cells, vec_clusters, t)

            if idx_reaction >= number_diff:
                if prol_rates[idx_reaction - number_diff] > 0:
                    number_cells[idx_reaction - number_diff] += 1
                    num_cell = gvec_pool(vec_tree, (idx_reaction - number_diff))

                    vec_tree[0, :, -2:] = np.zeros((2, 2))
                    vec_tree = gvec_tree(vec_tree, t, idx_reaction, num_cell)
                    vec_tree[1, :, -2:] = np.array([[1, 1], [idx_reaction - number_diff, idx_reaction - number_diff]])

                else:
                    number_cells[idx_reaction - number_diff] -= 1
                    num_cell = gvec_pool(vec_tree, (idx_reaction - number_diff))

                    vec_tree = gvec_tree(vec_tree, t, idx_reaction, num_cell)

            else:
                number_cells[M[idx_reaction, 0]] -= 1
                number_cells[M[idx_reaction, 1]] += 1
                num_cell = gvec_pool(vec_tree, M[idx_reaction, 0])

                vec_tree[0, :, :] = np.zeros((2, 1))
                vec_tree = gvec_tree(vec_tree, t, idx_reaction, num_cell)
                vec_tree[1, :, -1] = np.array([1, M[idx_reaction, 1]])

            st = np.squeeze(vec_tree[-1, :, :])
            vec_tree = vec_tree[1:, :, :]

            np.savetxt(f'./gillespie/store_{seed}.txt', st)

            print (
                f'iters: {iters}', 
                f't: {t:.3f}', 
                f'react_id: {idx_reaction}', 
                f'# cell: {number_cells}', 
                sep='\t'
            )
            iters += 1

    plt.show()