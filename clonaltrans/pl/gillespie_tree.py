import re
from collections import Counter, defaultdict
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import json
import ast
import os
import multiprocessing
from tqdm import tqdm
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
from itertools import combinations
import statsmodels.stats.multitest as smm

def get_div_distribution(gillespie_dir, cluster_names):
    div_path = os.path.join(gillespie_dir, 'res_div.txt')
    res_div = [[] for i in cluster_names[1:]]
    res_div = dict(zip(cluster_names[1:], res_div))
    num_trails = 0

    with open(div_path, 'r') as f:
        for idx, line in tqdm(enumerate(f)):
            line = line.split('\n')[0]
            line = json.loads(line)
            line = ast.literal_eval(line)
            del line['seed']
            num_trails += 1

            for key in line.keys():
                res_div[key].extend(line[key])
    
    return res_div, num_trails

def visualize_num_div(cluster_names, gillespie_dir, palette='tab20', save=False):
    res_div, num_trails = get_div_distribution(gillespie_dir, cluster_names)
    colors = get_hex_colors(palette)
    colors = colors * 2
    clone = gillespie_dir.split('/')[-2]

    for cid in range(1, len(cluster_names)):
        if res_div[cluster_names[cid]] != []:
            fig, axes = plt.subplots(1, 1, figsize=(6, 6))
            
            sns.histplot(res_div[cluster_names[cid]], ax=axes, color=colors[cid], bins=50)
            axes.set_title(cluster_names[cid], fontsize=14)
            axes.set_ylabel('# of seed trails', fontsize=13)
            axes.set_xlabel('# of divisions from a HSC', fontsize=13)

            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            # axes.set_xticks(fontsize=13)
            # axes.set_yticks(fontsize=13)

            plt.text(x=0.95, y=0.95, ha='right', va='top', color='black', fontsize=14,
                s=f'Mean # of divisions: ${np.mean(res_div[cluster_names[cid]]):.2f}$', transform=plt.gca().transAxes
            ) 
            plt.text(x=0.95, y=0.90, ha='right', va='top', color='black', fontsize=14,
                s=f'Total # of clones: ${len(res_div[cluster_names[cid]])} \;/\; {num_trails}$', transform=plt.gca().transAxes
            )

            if save:
                plt.savefig(f'./gdist_{clone}_{cluster_names[cid]}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def get_divisions(target_path, cluster_names):
    target_path_names = [get_cell_idx(i, types='name', cluster_names=cluster_names) for i in target_path]

    dict_counter = Counter(target_path_names)
    num_div = 0

    for cluster in dict_counter.keys():
        if cluster != target_path_names[-1]:
            num_div = num_div + dict_counter[cluster] - 1
    
    return num_div, target_path_names[-1]

def number_divisions(G, cluster_names):
    source = f'{cluster_names[0]}0'
    target = [f'{name}0' for name in cluster_names[1:]]
    paths = nx.single_source_shortest_path(G, source)
    
    paths_new = {}
    for item in target:
        if item in paths.keys():
            paths_new[item] = paths[item]

    num_divisions = defaultdict(list)

    for target in paths_new.keys():
        if not target.startswith('Death') and not target.startswith(source[:-1]):
            res, node_key = get_divisions(paths_new[target], cluster_names)
            num_divisions[node_key].append(res)

    return dict(num_divisions)

def multi_divisions(args):
    seed, cluster_names, gillespie_dir = args

    if os.path.exists(f'{gillespie_dir}/structure_{seed}.txt'):
        num_divisions = visualize_gtree(seed, cluster_names, gillespie_dir=gillespie_dir)
        num_divisions['seed'] = seed
        
        with open(f'{gillespie_dir}/res_div.txt', 'a') as f1:
            json.dump(str(num_divisions), f1)
            f1.write('\n')

def get_num_div(cluster_names, gillespie_dir='./gillespie'):
    try: multiprocessing.set_start_method('fork')
    except: pass

    if os.path.exists(f'{gillespie_dir}/res_div.txt'):
        os.remove(f'{gillespie_dir}/res_div.txt')

    num_boots = int(len(os.listdir(gillespie_dir)) / 3)
    clone = gillespie_dir.split('/')[-2:]
    print (f'Total # of gillespie trials for {clone}: {num_boots}')

    with multiprocessing.Pool(20) as pool:
        pool.map_async(
            multi_divisions, 
            [[seed, cluster_names, gillespie_dir] for seed in range(num_boots)]
        )
        
        pool.close()
        pool.join()

def get_hex_colors(colormap='tab20'):
    colormap = plt.get_cmap(colormap).colors
    return [to_hex(color) for color in colormap]

def get_cell_idx(node, types='index', cluster_names=None):
    length = 5

    for idx, item in enumerate(cluster_names):
        if item == node[:len(item)]:
            length = len(cluster_names[idx])
            break

    if types == 'index':
        return int(node[length:])
    if types == 'name':
        return node[:length]

def visualize_gtree(
    seed, 
    cluster_names, 
    palette: str ='tab20', 
    gillespie_dir: str ='./gillespie', 
    show: bool = False,
    save: any = False,
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
            line = json.loads(line)
            line = ast.literal_eval(line)

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
        
        pos = graphviz_layout(G, prog='dot')
        colors = [cluster_colors[G.nodes[node]['cluster']] for node in G.nodes()]
        labels = {node:get_cell_idx(node, types='index', cluster_names=cluster_names) for node in G.nodes()}

        nx.draw(
            G, pos, ax=axes, node_color=colors, labels=labels, 
            with_labels=True, node_size=1500, font_size=25
        )
        nx.draw_networkx_edge_labels(
            G, 
            pos, 
            time_stamps, 
            font_size=25, 
            rotate=False, 
            ax=axes
        )

        patches = [plt.plot([], [], marker='o', ls='', color=color, markersize=13)[0] for color in cluster_colors.values()] 
        plt.legend(patches, cluster_colors.keys(), bbox_to_anchor=(1, 0.9), fontsize=23, frameon=False)
        plt.title(f'Tree structure of seed {seed}', fontsize=23)

        if save:
            plt.savefig(f'./gillespie_tree_{seed}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

    else:
        return number_divisions(G, cluster_names)

def mean_division_to_first(mean_div_distributions, palette='tab20', save=False):
    df = pd.DataFrame(mean_div_distributions)
    index = [f'Clone {i}' for i in range(len(mean_div_distributions))]
    index[-1] = 'Clone BG'
    df.index = index

    color = get_hex_colors(palette)
    color = color * 2
    colors = [color[i + 1] for i, name in enumerate(df.columns)]
    ax = df.plot(kind='bar', stacked=False, figsize=(10, 5), width=0.8, color=colors)

    plt.legend(bbox_to_anchor=(1, 1.04), frameon=False)
    plt.xticks(rotation=90)
    plt.ylabel('# of divisions')
    plt.title('Mean # of divisions needed to produced the first progeny') 

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def succeed_trails_to_first(len_div_distributions, num_trails=None, palette='tab20', save=False):
    df = pd.DataFrame(len_div_distributions)
    df = df.div(np.max(df.values, axis=1), axis=0) if num_trails is None else df.div(num_trails, axis=0)

    # df = df.applymap(lambda x: np.log(x + 1))

    index = [f'Clone {i}' for i in range(len(len_div_distributions))]
    index[-1] = 'Clone BG'
    df.index = index

    color = get_hex_colors(palette)
    color = color * 2
    colors = [color[i + 1] for i, name in enumerate(df.columns)]
    ax = df.plot(kind='bar', stacked=False, figsize=(10, 5), width=0.8, color=colors)

    plt.legend(bbox_to_anchor=(1, 1.04), frameon=False)
    plt.ylabel('# of trails')
    plt.title('Potency preferance of HSCs for each population')

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def clone_dist_diff_plot(div_distributions, ref_model, correct_method='holm', save=False):
    num_clone, num_pop = ref_model.N.shape[1], ref_model.N.shape[2]
    stats_tests = np.zeros((int(num_clone * (num_clone - 1) / 2), num_pop - 1))
    stats_tests[stats_tests == 0] = 'nan'

    count, index = 0, []
    for (c1, c2) in combinations(range(num_clone), 2):
        for idx_pop, pop in enumerate(div_distributions[0].keys()):
                div_c1 = div_distributions[c1][pop]
                div_c2 = div_distributions[c2][pop]

                if len(div_c1) >= 3 and len(div_c2) >= 3:
                    _, shapiro_p_c1 = stats.shapiro(div_c1)
                    _, shapiro_p_c2 = stats.shapiro(div_c2)

                    if shapiro_p_c1 > 0.05 and shapiro_p_c2 > 0.05:
                        _, paired_p = stats.ttest_ind(div_c1, div_c2) 
                        stats_tests[count, idx_pop] = paired_p
                    else:
                        _, wilcox_p = stats.mannwhitneyu(div_c1, div_c2)
                        stats_tests[count, idx_pop] = wilcox_p
        count += 1

        if c1 == num_clone - 1:
            c1 = 'BG'
        if c2 == num_clone - 1:
            c2 = 'BG'
        index.append(f'Clone {c1} / {c2}')

    adjusted_p_values = []
    for i in range(stats_tests.shape[1]):
        adjusted_p_values.append(smm.multipletests(stats_tests[:, i], method=correct_method)[1])

    fig, axes = plt.subplots(figsize=(50, 15))
    adjusted_p_values = np.stack(adjusted_p_values)
    adjusted_p_values[adjusted_p_values > 0.05] = np.nan

    df = pd.DataFrame(data=-np.log10(adjusted_p_values), index=list(div_distributions[0].keys()), columns=index)
    ax = sns.heatmap(df, annot=False, linewidths=.5, cmap='viridis')
    plt.title('$-log_{10}$ p-values of # of divisions needed across meta-clones & populations')
    plt.xticks(rotation=90)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')