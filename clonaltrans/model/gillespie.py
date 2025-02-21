import numpy as np
import random
import os
import pandas as pd
import json
import copy
from utils import set_seed

def gillespie_rates(matrix_K, L, init=False):
    diff_rates = copy.deepcopy(matrix_K) # (num_pops, num_pops)
    np.fill_diagonal(diff_rates, 0)
    diff_rates = diff_rates[L != 0]

    prol_rates = copy.deepcopy(matrix_K)
    prol_rates = np.diag(prol_rates)

    # Possible reactions for the gillespie
    rates = np.concatenate((diff_rates, np.abs(prol_rates)))

    if init:
        # Directions of differentiation matrix
        directions = np.where(L > 0)

        M = np.zeros((len(diff_rates), 2), dtype=int)
        M[:, 0], M[:, 1] = directions[0], directions[1]
        return M, rates

    else:
        return prol_rates, rates

def get_propensity_vec(rates, number_cells, vec_clusters):
    #* the total propensity score & probability of each reaction
    propensity = rates * np.array([number_cells[0, i] for i in vec_clusters])
    propensity_total = np.sum(propensity)
    return propensity, propensity_total

def gillespie_module(propensity, propensity_total, t, t_cutoff):
    r = propensity / propensity_total
    
    # randomly choose when next reaction will occur and calculate time increment
    extract_time = random.random()
    delta_t = -np.log(extract_time) / propensity_total 

    delta_t = np.max([delta_t, t_cutoff])
    t += delta_t 

    # randomly choose which reaction will occur
    extract_reaction = random.random()
    aux = extract_reaction - np.cumsum(r)
    aux[aux >= 0] = -10

    idx_reaction = np.argmax(aux)
    return idx_reaction, t

def gillespie_main(
    seed, 
    K_total, # (num_time_points, num_pops, num_pops)
    time_all,
    sampled_key,
    cluster_names,
    gillespie_dir,
    t_cutoff,
    L
):
    set_seed(seed)

    if os.path.exists(f'{gillespie_dir}/occurred_{seed}.csv'):
        os.remove(f'{gillespie_dir}/occurred_{seed}.csv')
    if os.path.exists(f'{gillespie_dir}/structure_{seed}.txt'):
        os.remove(f'{gillespie_dir}/structure_{seed}.txt')
    if os.path.exists(f'{gillespie_dir}/num_cells_{seed}.csv'):
        os.remove(f'{gillespie_dir}/num_cells_{seed}.csv')

    M, rates = gillespie_rates(K_total[0], L, init=True)
    number_diff_rates = M.shape[0]
    occurred_reactions = np.zeros(len(rates))

    #* Index of outbound clusters corresponding to the reactions
    #* M[:, 0] is the outbound of differentiation reactions
    #* np.arange(0, len(cluster_names)) is the outbound of proliferation reactions
    vec_clusters = np.concatenate((M[:, 0], np.arange(0, len(cluster_names)))) 

    #* Random initialize the first cell based on the initial distribution of the cells
    idx_init_cluster = np.where(cluster_names == sampled_key)[0][0]
    number_cells = np.zeros((1, len(cluster_names)))
    number_cells[0, idx_init_cluster] = 1

    values = [[] for i in range(len(cluster_names))]
    values[idx_init_cluster].append(0)
    
    cell_ids = dict(zip(cluster_names, values))
    cell_ids['Descriptions'] = []

    cell_ids_max = np.zeros(len(cluster_names)) - 1
    cell_ids_max[idx_init_cluster] = 0

    t = 0
    list_number_cells = number_cells.copy()

    while np.sum(number_cells) > 0 and t <= time_all[-1]:
        propensity, propensity_total = get_propensity_vec(rates, number_cells, vec_clusters)

        if propensity_total == 0:
            break

        idx_reaction, t = gillespie_module(propensity, propensity_total, t, t_cutoff)

        occurred_reactions[idx_reaction] += 1

        prol_rates, rates = gillespie_rates(K_total[np.abs(time_all - t).argmin()], L, init=False)

        if idx_reaction >= number_diff_rates:
            total_id = idx_reaction - number_diff_rates

            if prol_rates[total_id] > 0:
                number_cells[0, total_id] += 1

                loc_id = np.random.randint(0, len(cell_ids[cluster_names[total_id]]))
                idx_cell = cell_ids[cluster_names[total_id]][loc_id]

                cell_ids_max[total_id] += 1
                cell_ids[cluster_names[total_id]].append(int(cell_ids_max[total_id]))
                cell_ids_max[total_id] += 1
                cell_ids[cluster_names[total_id]].append(int(cell_ids_max[total_id]))

                cell_ids['Descriptions'] = [cluster_names[total_id], idx_cell, 'Prol', [int(cell_ids_max[total_id]) - 1, int(cell_ids_max[total_id])], np.round(t, 3)]
                cell_ids[cluster_names[total_id]].remove(idx_cell)

            if prol_rates[total_id] < 0:
                number_cells[0, total_id] -= 1

                loc_id = np.random.randint(0, len(cell_ids[cluster_names[total_id]]))
                idx_cell = cell_ids[cluster_names[total_id]][loc_id]

                cell_ids['Descriptions'] = [cluster_names[total_id], idx_cell, 'Apop', np.round(t, 3)]
                cell_ids[cluster_names[total_id]].remove(idx_cell)
            
        else:
            out_id, in_id = M[idx_reaction, 0], M[idx_reaction, 1]

            number_cells[0, out_id] -= 1
            number_cells[0, in_id] += 1

            loc_id = np.random.randint(0, len(cell_ids[cluster_names[out_id]]))
            idx_cell = cell_ids[cluster_names[out_id]][loc_id]
            cell_ids[cluster_names[out_id]].remove(idx_cell)

            cell_ids_max[in_id] += 1
            cell_ids[cluster_names[in_id]].append(int(cell_ids_max[in_id]))
            cell_ids['Descriptions'] = [cluster_names[out_id], idx_cell, 'Diff', [cluster_names[in_id], int(cell_ids_max[in_id])], np.round(t, 3)]

        list_number_cells = np.concatenate([list_number_cells, number_cells])

        with open(f'{gillespie_dir}/structure_{seed}.txt', 'a') as f1:
            json.dump(str(cell_ids['Descriptions']), f1)
            f1.write('\n')

    occurred_reactions = pd.DataFrame(index=range(len(occurred_reactions)), columns=['# of reactions'], data=occurred_reactions)
    occurred_reactions.to_csv(f'{gillespie_dir}/occurred_{seed}.csv')

    list_number_cells = pd.DataFrame(index=range(len(list_number_cells)), columns=cluster_names, data=list_number_cells)
    list_number_cells['Total counts'] = np.sum(list_number_cells.values, axis=1)
    list_number_cells.to_csv(f'{gillespie_dir}/num_cells_{seed}.csv')