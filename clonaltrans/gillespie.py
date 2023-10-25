import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import json
import ast
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from natsort import natsorted
from matplotlib.colors import to_hex
import multiprocessing
from tqdm import tqdm
import re
from collections import Counter, defaultdict
from scipy import stats
import copy
import torch

def get_hex_colors(colormap='tab20'):
    colormap = plt.get_cmap(colormap).colors
    return [to_hex(color) for color in colormap]

def gillespie_rates(matrix_K, init=False):
    diff_rates = copy.deepcopy(matrix_K)
    np.fill_diagonal(diff_rates, 0)
    diff_rates = diff_rates.flatten()[diff_rates.flatten().nonzero()]

    prol_rates = copy.deepcopy(matrix_K)
    prol_rates = np.diag(prol_rates)

    # Possible reactions for the gillespie
    rates = np.concatenate((diff_rates, np.abs(prol_rates)))

    if init:
        # Directions of differentiation matrix
        get_index = copy.deepcopy(matrix_K)
        np.fill_diagonal(get_index, 0)
        directions = np.where(get_index > 0)

        M = np.zeros((len(diff_rates), 2), dtype=int)
        M[:, 0], M[:, 1] = directions[0], directions[1]

        return M, rates, len(diff_rates)

    else:
        return prol_rates, rates

def gillespie_module(rates, number_cells, vec_clusters, t):
    #* the total propensity score & probability of each reaction
    propensity = rates * np.array([number_cells[0, i] for i in vec_clusters])
    rates_vec = np.sum(propensity)
    r = propensity / rates_vec
    
    # randomly choose when next reaction will occur and calculate time increment
    extract_time = random.random()
    delta_t = -np.log(extract_time) / rates_vec 

    delta_t = np.max([delta_t, 5e-5])
    t += delta_t 

    # randomly choose which reaction will occur
    extract_reaction = random.random()
    aux = extract_reaction - np.cumsum(r)
    aux[aux >= 0] = -10

    idx_reaction = np.argmax(aux)
    return idx_reaction, t

def gillespie_rates_prepare(model, index_clone):
    print (f'---> Preparing candidate transition rates for meta-clone {index_clone}.')
    time_all = np.arange(model.t_observed[0].cpu(), model.t_observed[-1].cpu(), 0.1) 
    K_total = []

    for time in time_all:
        K_total.append(model.get_matrix_K(model.config.K_type, eval=True, tpoint=torch.Tensor([time]).to('cpu'))[index_clone].detach().cpu().numpy())
    
    return np.stack(K_total), time_all

def gillespie_main(
    seed, 
    data_dir,
    index_clone, 
    K_total,
    time_all
):
    random.seed(seed)
    np.random.seed(seed)

    gillespie_dir = os.path.join(data_dir, f'models/gillespie/clone_{index_clone}')
    cluster_names = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))['populations'][:K_total.shape[2]]

    if not os.path.exists(gillespie_dir):
        os.mkdir(gillespie_dir)
    if os.path.exists(f'./{gillespie_dir}/occurred_{seed}.csv'):
        os.remove(f'./{gillespie_dir}/occurred_{seed}.csv')
    if os.path.exists(f'{gillespie_dir}/structure_{seed}.txt'):
        os.remove(f'{gillespie_dir}/structure_{seed}.txt')
    if os.path.exists(f'./{gillespie_dir}/num_cells_{seed}.csv'):
        os.remove(f'./{gillespie_dir}/num_cells_{seed}.csv')
    print (f'---> Existing files are overwrittened for seed {seed}.')

    M, rates, number_diff_rates = gillespie_rates(K_total[0], init=True)
    occurred_reactions = np.zeros(len(rates))

    #* Index of outbound clusters corresponding to the reactions
    vec_clusters = np.concatenate((M[:, 0], np.arange(0, len(cluster_names)))) 

    number_cells = np.zeros((1, len(cluster_names)))
    number_cells[0, 0] = 1 #* Start with one stem cell *#

    keys = [i for i in cluster_names]
    keys.append(-1)
    values = [[] for i in range(len(cluster_names))]
    values.append([])
    values[0].append(0)
    cell_ids = dict(zip(keys, values))

    cell_ids_max = np.zeros(len(cluster_names)) - 1
    cell_ids_max[0] = 0

    t = 0
    list_number_cells = number_cells.copy()
    print ('---> Simulations started.')

    while np.sum(number_cells) > 0 and t <= time_all[-1]:
        idx_reaction, t = gillespie_module(rates, number_cells, vec_clusters, t)
        occurred_reactions[idx_reaction] += 1

        prol_rates, rates = gillespie_rates(K_total[np.abs(time_all - t).argmin()], init=False)

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

                cell_ids[-1] = [cluster_names[total_id], idx_cell, 'Prol', [int(cell_ids_max[total_id]) - 1, int(cell_ids_max[total_id])], np.round(t, 3)]
                cell_ids[cluster_names[total_id]].remove(idx_cell)

            if prol_rates[total_id] < 0:
                number_cells[0, total_id] -= 1

                loc_id = np.random.randint(0, len(cell_ids[cluster_names[total_id]]))
                idx_cell = cell_ids[cluster_names[total_id]][loc_id]

                cell_ids[-1] = [cluster_names[total_id], idx_cell, 'Apop', np.round(t, 3)]
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
            cell_ids[-1] = [cluster_names[out_id], idx_cell, 'Diff', [cluster_names[in_id], int(cell_ids_max[in_id])], np.round(t, 3)]

        list_number_cells = np.concatenate([list_number_cells, number_cells])

        with open(f'{gillespie_dir}/structure_{seed}.txt', 'a') as f1:
            json.dump(str(cell_ids[-1]), f1)
            f1.write('\n')
    
    occurred_reactions = pd.DataFrame(index=range(len(occurred_reactions)), columns=['# of reactions'], data=occurred_reactions)
    occurred_reactions.to_csv(f'./{gillespie_dir}/occurred_{seed}.csv')

    list_number_cells = pd.DataFrame(index=range(len(list_number_cells)), columns=keys[:-1], data=list_number_cells)
    list_number_cells['Total counts'] = np.sum(list_number_cells.values, axis=1)
    list_number_cells.to_csv(f'./{gillespie_dir}/num_cells_{seed}.csv')

def gplot_trees(
    seed, 
    cluster_names: list = ['A', 'B', 'C', 'D', 'E'],
    palette: str ='tab20', 
    gillespie_dir: str ='./gillespie', 
    show: bool = False
):
    colors = get_hex_colors(palette)
    cluster_colors = dict(zip(cluster_names, colors[:len(cluster_names)]))
    cluster_colors['Death'] = 'black'
    time_stamps = {}

    G = nx.DiGraph()
    G.add_node(f'{cluster_names[0]}0', cluster=cluster_names[0])

    num_of_deaths = 0
    with open(f'{gillespie_dir}/structure_{seed}.txt', 'r') as f:
        for iters, line in enumerate(f):
            # line = line.split('\n')[0]
            line = json.loads(line)
            line = ast.literal_eval(line)
            # line = line[-1]

            if line[2] == 'Prol':
                G.add_node(line[0] + str(line[3][0]), cluster=line[0])
                G.add_edge(line[0] + str(line[1]), line[0] + str(line[3][0]))
                time_stamps[(line[0] + str(line[1]), line[0] + str(line[3][0]))] = f't {iters}'

                G.add_node(line[0] + str(line[3][1]), cluster=line[0])
                G.add_edge(line[0] + str(line[1]), line[0] + str(line[3][1]))
                time_stamps[(line[0] + str(line[1]), line[0] + str(line[3][1]))] = f't {iters}'

            if line[2] == 'Diff':
                G.add_node(line[3][0] + str(line[3][1]), cluster=line[3][0])
                G.add_edge(line[0] + str(line[1]), line[3][0] + str(line[3][1]))
                time_stamps[(line[0] + str(line[1]), line[3][0] + str(line[3][1]))] = f't {iters}'

            if line[2] == 'Apop':
                G.add_node(f'Death{num_of_deaths}', cluster='Death')
                G.add_edge(line[0] + str(line[1]), f'Death{num_of_deaths}')
                time_stamps[(line[0] + str(line[1]), f'Death{num_of_deaths}')] = f't {iters}'
                num_of_deaths += 1

    if show:
        fig, axes = plt.subplots(1, 1, figsize=(15, 15))
        # plt.title(f'Tree structure of seed {seed}', fontsize=12)
        pos = graphviz_layout(G, prog='dot')
        colors = [cluster_colors[G.nodes[node]['cluster']] for node in G.nodes()]
        labels = {node:get_cell_idx(node, types='index') for node in G.nodes()}

        # nx.draw(
        #     G, pos, ax=axes, node_color=colors, labels=labels, 
        #     with_labels=True, node_size=1500, font_size=25
        # )
        nx.draw_networkx_edge_labels(
            G, 
            pos, 
            time_stamps, 
            font_size=25, 
            rotate=False, 
            ax=axes
        )

        patches = [plt.plot([], [], marker='o', ls='', color=color, markersize=13)[0] for color in cluster_colors.values()] 
        plt.legend(patches, cluster_colors.keys(), loc='center right', fontsize=23, frameon=False)
        # plt.show()
        plt.savefig(f'./gillespie_example_{seed}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

    else:
        num_divisions = number_divisions(G, cluster_names)
        return num_divisions

def get_cell_idx(node, types='index'):
    numbers = re.findall(r'\d+', node)[0]

    if types == 'index':
        return int(numbers)
    if types == 'name':
        return node[:-len(numbers)]

def get_divisions(target_path):
    target_path_names = [get_cell_idx(i, types='name') for i in target_path]

    dict_counter = Counter(target_path_names)
    num_div = 0

    for cluster in dict_counter.keys():
        if cluster != target_path_names[-1]:
            num_div = num_div + dict_counter[cluster] - 1
    
    return num_div, target_path_names[-1]

def number_divisions(G, cluster_names):
    source = f'{cluster_names[0]}0'
    target = [f'{name}0' for name in cluster_names[1:]]
    # paths = nx.shortest_path(G, source, None)
    paths = nx.single_source_shortest_path(G, source)
    
    paths_new = {}
    for item in target:
        if item in paths.keys():
            paths_new[item] = paths[item]

    num_divisions = defaultdict(list)

    for target in paths_new.keys():
        if not target.startswith('Death') and not target.startswith(source[:-1]):
            res, node_key = get_divisions(paths_new[target])
            num_divisions[node_key].append(res)
    
    return dict(num_divisions)

def get_num_divisions(cluster_names, gillespie_dir='./gillespie'):
    multiprocessing.set_start_method('fork')
    if os.path.exists(f'{gillespie_dir}/res_div.txt'):
        os.remove(f'{gillespie_dir}/res_div.txt')

    with multiprocessing.Pool(40) as pool:
        pool.map_async(
            multi_divisions, 
            [[seed, cluster_names, gillespie_dir] for seed in range(5000)]
        )
        
        pool.close()
        pool.join()

def multi_divisions(args):
    seed, cluster_names, gillespie_dir = args
    if os.path.exists(f'{gillespie_dir}/structure_{seed}.txt'):
        num_divisions = gplot_trees(seed, cluster_names, gillespie_dir=gillespie_dir)
        
        with open(f'{gillespie_dir}/res_div.txt', 'a') as f1:
            json.dump(str(num_divisions), f1)
            f1.write('\n')

def get_div_distribution(div_path, cluster_names):
    res_div = [[] for i in cluster_names[1:]]
    res_div = dict(zip(cluster_names[1:], res_div))

    with open(div_path, 'r') as f:
        for idx, line in tqdm(enumerate(f)):
            line = line.split('\n')[0]
            line = json.loads(line)
            line = ast.literal_eval(line)

            for key in line.keys():
                res_div[key].extend(line[key])
    
    return res_div

def gplot_num_divisions(div_path, cluster_names, palette='tab20'):
    res_div = get_div_distribution(div_path, cluster_names)

    colors = get_hex_colors(palette)
    num_clusters = len(cluster_names) - 1

    for cid in range(num_clusters):
        if res_div[cluster_names[cid + 1]] != []:
            fig, axes = plt.subplots(1, 1, figsize=(6, 6))
            sns.histplot(res_div[cluster_names[cid + 1]], ax=axes, color=colors[cid + 1], binwidth=1)
            axes.set_title(cluster_names[cid + 1], fontsize=14)
            axes.set_ylabel('# of seed trails', fontsize=13)
            axes.set_xlabel('# of divisions happened between a HSC', fontsize=13)

            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            # axes.set_xticks(fontsize=13)
            # axes.set_yticks(fontsize=13)

            plt.text(x=0.95, y=0.95, ha='right', va='top', color='black', fontsize=14,
                s=f'Mean # of divisions: ${np.mean(res_div[cluster_names[cid + 1]]):.2f}$', transform=plt.gca().transAxes
            ) 
            plt.text(x=0.95, y=0.90, ha='right', va='top', color='black', fontsize=14,
                s=f'Total # of clones: ${len(res_div[cluster_names[cid + 1]])} \;/\; 5000$', transform=plt.gca().transAxes
            )
            plt.savefig(f'./gillespie_div_clone0_{cluster_names[cid + 1]}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def div_diffclones(div_distributions, ref_model, save='clonediffdivs'):
    overall_res = np.ones((15, ref_model.N.shape[2] - 1))
    overall_res[overall_res == 1] = 'NaN'

    count, index = 0, []
    for idx_c1 in range(ref_model.N.shape[1] - 2):
        for idx_c2 in range(idx_c1 + 1, ref_model.N.shape[1] - 1):
            index.append(f'Clone {idx_c1} / {idx_c2}')
            # print (f'Test clone {idx_c1, idx_c2}.')

            for idx_pop in range(len(div_distributions[0].keys())):
                    pop = list(div_distributions[0].keys())[idx_pop]
                    div_c1 = div_distributions[idx_c1][pop]
                    div_c2 = div_distributions[idx_c2][pop]

                    if div_c1 != [] and div_c2 != []:
                        _, shapiro_p_c1 = stats.shapiro(div_c1)
                        _, shapiro_p_c2 = stats.shapiro(div_c2)

                        if shapiro_p_c1 > 0.05 and shapiro_p_c2 > 0.05:
                            print (idx_c1, idx_c2, pop)
                            _, paired_p = stats.ttest_ind(div_c1, div_c2) 
                            overall_res[count, idx_pop] = paired_p
                        else:
                            _, wilcox_p = stats.mannwhitneyu(div_c1, div_c2)
                            overall_res[count, idx_pop] = wilcox_p
            count += 1
    
    anno = pd.read_csv(os.path.join(ref_model.data_dir, 'annotations.csv'))
    graph = pd.read_csv(os.path.join(ref_model.data_dir, 'graph_table.csv'), index_col=0)
    np.fill_diagonal(graph.values, 1)

    import statsmodels.stats.multitest as smm
    adjusted_p_values = []
    for i in range(overall_res.shape[1]):
        adjusted_p_values.append(smm.multipletests(overall_res[:, i], method='holm')[1])
    print(np.stack(adjusted_p_values).shape)

    fig, axes = plt.subplots(figsize=(18, 7))
    df = pd.DataFrame(data=-np.log10(np.stack(adjusted_p_values)), index=list(div_distributions[0].keys()), columns=index)
    sns.heatmap(df, annot=False, linewidths=.5, cmap='viridis', vmin=0, vmax=200)
    plt.title('$-log_{10}$ p-values of # of divisions needed across meta-clones & populations')
    plt.xticks(rotation=30)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

class MultiProcess():
    def __init__(self, data_dir, index_clone=0, num_gpus=10, num_boots=5000, K_total=None, time_all=None) -> None:
        super(MultiProcess, self).__init__()
        self.num_gpus = num_gpus
        self.num_boots = num_boots

        self.data_dir = data_dir
        self.index_clone = index_clone

        self.K_total, self.time_all = K_total, time_all

    def bootstart(self):
        multiprocessing.set_start_method('spawn')

        pbar = tqdm(range(self.num_boots))
        print ('---> Multiprocessing.')

        with multiprocessing.Pool(self.num_gpus) as pool:
            for epoch in pbar:
                for res in pool.imap_unordered(
                    self.process, 
                    self.get_buffer()
                ):
                    pass

    def get_buffer(self):
        buffer = []
        for i in range(self.num_boots):
            buffer.append([i, self.data_dir, self.index_clone, self.K_total, self.time_all])
        return buffer

    def process(self, args):
        seed, data_dir, index_clone, K_total, time_all = args

        gillespie_main(
            seed,
            data_dir, 
            index_clone,
            K_total,
            time_all
        )