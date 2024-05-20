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
from itertools import combinations, product
import statsmodels.stats.multitest as smm
from .metrics import get_clustered_heatmap
from .base import get_subplot_dimensions
import copy
from natsort import natsorted

def get_div_distribution(gillespie_dir, cluster_names):
    div_path = os.path.join(gillespie_dir, 'res_div.txt')
    res_div = [[] for i in cluster_names[1:]]
    res_div = dict(zip(cluster_names[1:], res_div))
    num_trails = 0

    with open(div_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.split('\n')[0]
            line = json.loads(line)
            line = ast.literal_eval(line)
            del line['seed']
            num_trails += 1

            for key in line.keys():
                res_div[key].extend(line[key])
    
    return res_div, num_trails

def visualize_num_div(
    cluster_names,
    gillespie_dir, 
    clone_id=5,
    palette='tab20', 
    save=False
):
    res_div, num_trails = get_div_distribution(gillespie_dir, cluster_names)
    colors = get_hex_colors(palette)
    colors = colors * 2
    clone = gillespie_dir.split('/')[-1]

    # rows, cols, figsize = get_subplot_dimensions(len(cluster_names) - 1, max_cols=max_cols, fig_height_per_row=4)
    # fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] + 5, figsize[1] + 10))
    # fig.suptitle(f'Quantitative lens of division summary given a HSC_MPP for {os.path.split(gillespie_dir)[1]}', fontsize=22)
    # plt.subplots_adjust(top=0.93)

    for cid in range(1, len(cluster_names)):
        if res_div[cluster_names[cid]] != []:
            # ax_loc = axes[(cid - 1) // cols][(cid - 1) % cols] if rows > 1 else axes[cid - 1]
            
            # sns.histplot(res_div[cluster_names[cid]], ax=ax_loc, color=colors[cid])
            # ax_loc.set_title(cluster_names[cid], fontsize=20)

            # ax_loc.axvline(
            #     np.round(np.mean(res_div[cluster_names[cid]]), 2), 
            #     linestyle='--', 
            #     color='black', 
            #     linewidth=3,
            #     ymax=0.7
            # )

            # axes[(cid - 1) // cols][(cid - 1) % cols].set_ylabel('# of trails', fontsize=18)
            # axes[(cid - 1) // cols][(cid - 1) % cols].set_xlabel('# of divisions', fontsize=18)

            # if (cid - 1) % cols != 0:
                # axes[(cid - 1) // cols][(cid - 1) % cols].set_ylabel('')

            ax = sns.histplot(res_div[cluster_names[cid]], color=colors[cid])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=18)

            fig = ax.get_figure()
            fig.set_figwidth(6) 
            fig.set_figheight(6)

            plt.title(f'{cluster_names[cid]} (Meta-clone {clone_id})', fontsize=20)
            plt.xlabel('# of divisions', fontsize=18)
            plt.ylabel('# of trails', fontsize=18)
            plt.text(x=0.93, y=0.92, ha='right', va='top', color='black', fontsize=18,
                s=f'Mean: ${np.round(np.mean(res_div[cluster_names[cid]]), 2)}$', transform=ax.transAxes
            )

            if save:
                plt.savefig(f'./gdist_{clone}_{cluster_names[cid]}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')
                plt.close()

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
        plt.legend(patches, cluster_colors.keys(), bbox_to_anchor=(1, 0.95), fontsize=23, frameon=False)
        plt.title(f'Simulated differentiation structure of seed {seed} | {os.path.split(gillespie_dir)[1]}', fontsize=23)

        if save:
            plt.savefig(f'./gillespie_tree_{seed}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

    else:
        return number_divisions(G, cluster_names)

def mean_division_to_first(mean_div_distributions, palette='tab20', save=False):
    df = pd.DataFrame(mean_div_distributions)
    index = [f'{i}' for i in range(len(mean_div_distributions))]
    index[-1] = 'BG'
    df.index = index

    color = get_hex_colors(palette)
    color = color * 2
    colors = [color[i + 1] for i, name in enumerate(df.columns)]
    ax = df.plot(kind='bar', stacked=False, figsize=(40, 10), width=0.9, color=colors)

    plt.legend(bbox_to_anchor=(0.95, -0.05), frameon=False, fontsize=28, ncol=7)
    plt.xticks(rotation=0, fontsize=28)
    plt.yticks(fontsize=28)
    plt.title('Mean # of divisions needed for producing the first progeny given a HSC_MPP for each meta-clone', fontsize=32, pad=5) 
    plt.tick_params(axis='both', which='both', width=2, length=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def succeed_trails_to_first(len_div_distributions, num_trails=None, palette='tab20', save=False):
    df = pd.DataFrame(len_div_distributions)
    # df = df.div(np.max(df.values, axis=1), axis=0) if num_trails is None else df.div(num_trails, axis=0) * 100
    # df = df.map(lambda x: np.log(x + 1))

    index = [f'{i}' for i in range(len(len_div_distributions))]
    index[-1] = 'BG'
    df.index = index

    color = get_hex_colors(palette)
    color = color * 2
    colors = [color[i + 1] for i, name in enumerate(df.columns)]
    ax = df.plot(kind='bar', stacked=False, figsize=(40, 10), width=0.9, color=colors)

    plt.legend(bbox_to_anchor=(0.9, -0.05), frameon=False, fontsize=30, ncol=5)
    plt.xticks(rotation=0, fontsize=30)
    plt.yticks([0, 200, 400, 600, 800, 1000], fontsize=30)
    plt.ylabel(f'# of simulation trails', fontsize=40)
    plt.title('Potency preferance for each meta-clone given a HSC_MPP', fontsize=40, pad=0)
    plt.tick_params(axis='both', which='both', width=2, length=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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

                stats_tests[count, idx_pop] = np.mean(div_c1) / np.mean(div_c2) if len(div_c1) > 10 and len(div_c2) > 10 else np.nan

                # if len(div_c1) >= 3 and len(div_c2) >= 3:
                #     _, shapiro_p_c1 = stats.shapiro(div_c1)
                #     _, shapiro_p_c2 = stats.shapiro(div_c2)

                #     if shapiro_p_c1 > 0.05 and shapiro_p_c2 > 0.05:
                #         _, paired_p = stats.ttest_ind(div_c1, div_c2) 
                #         stats_tests[count, idx_pop] = paired_p
                #     else:
                #         _, wilcox_p = stats.mannwhitneyu(div_c1, div_c2)
                #         stats_tests[count, idx_pop] = wilcox_p
        count += 1

        if c1 == num_clone - 1:
            c1 = 'BG'
        if c2 == num_clone - 1:
            c2 = 'BG'
        index.append(f'{c1} / {c2}')

    # adjusted_p_values = []
    # for i in range(stats_tests.shape[1]):
    #     adjusted_p_values.append(smm.multipletests(stats_tests[:, i], method=correct_method)[1])

    fig, axes = plt.subplots(figsize=(55, 12))
    # adjusted_p_values = np.stack(adjusted_p_values)
    # adjusted_p_values[adjusted_p_values > 0.01] = np.nan

    # df = pd.DataFrame(data=-np.log10(adjusted_p_values), index=list(div_distributions[0].keys()), columns=index)
    df = pd.DataFrame(data=stats_tests.T, index=list(div_distributions[0].keys()), columns=index)
    # df = df.filter(like='BG')
    # df = get_clustered_heatmap(df)

    ax = sns.heatmap(df, annot=False, linewidths=.1, cmap='coolwarm', xticklabels=True, yticklabels=True, cbar=True, vmin=0, vmax=3)
    # plt.title('$-log_{10}$ p-values of # of divisions', fontsize=18, pad=15)
    plt.title('Fold change of mean # of division events needed', fontsize=30, pad=15)
    plt.xticks(fontsize=25, rotation=90)
    plt.yticks(fontsize=25)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')

def find_successive_ones(df, row_index):
    cols_with_one = df.columns[df.loc[row_index] == 1].tolist()
    result = copy.deepcopy(cols_with_one)
    
    while cols_with_one:
        next_cols_with_one = []
        for item in cols_with_one:
            next_cols_with_one.extend(df.columns[df.loc[item] == 1].tolist())
        
        result.extend(next_cols_with_one)
        cols_with_one = next_cols_with_one
    
    return natsorted(list(set(result)))

def get_descendents(model, label, gillespie_dir):
    data = []
    result_path = os.path.join(gillespie_dir, 'res_div.txt')
    paga = pd.read_csv(os.path.join(
        model.config['data_loader']['args']['data_dir'], model.config['data_loader']['args']['graphs'],
    ), index_col=0).astype(np.int32)
    
    with open(result_path, 'r') as f:
        for line in f.readlines():            
            if not line:
                continue
            
            line = line.strip()
            line = ast.literal_eval(line)
            line = eval(line)

            if label in line.keys():
                data.append(line)
    
    descendents = find_successive_ones(paga, label)
    
    return data, descendents

def get_fate_prob(model, cluster_names, gillespie_dir):
    aggre = dict()
        
    for directory in natsorted(os.listdir(gillespie_dir)):
        if directory.startswith('clone'):
            gillespie_dir_clones = os.path.join(gillespie_dir, directory)
            aggre[directory] = {}

            distribution, counts = get_div_distribution(gillespie_dir_clones, cluster_names)
            for key in distribution.keys():
                distribution[key] = len(distribution[key]) / counts

            aggre[directory][cluster_names[0]] = distribution
            # aggre[directory][cluster_names[0]] = [list(distribution.keys()), list(distribution.values())]

            for label in cluster_names[1:]:
                data, des = get_descendents(model, label, gillespie_dir=gillespie_dir_clones)

                if len(des) >= 2:
                    num = np.zeros(len(des))
                    for idx, child in enumerate(des):
                        num[idx] += np.sum([True if child in trail.keys() else False for trail in data])

                    aggre[directory][label] = dict(zip(des, list(num / len(data)) if len(data) != 0 else list(num)))
                    # aggre[directory][label] = [des, list(num / len(data)) if len(data) != 0 else list(num)]

    return aggre

def get_fate_prec(aggre):
    res = dict()
    for clone, pop in product(aggre.keys(), aggre[list(aggre.keys())[0]].keys()):
        if pop not in res.keys():
            res[pop] = [list(aggre[clone][pop].keys()), [list(aggre[clone][pop].values())]]
        else:
            res[pop][1].append(list(aggre[clone][pop].values()))
    
    return res

def pl_fate_prob(aggre, label, logy=False, palette='tab20', save=False):
    results = get_fate_prec(aggre)
    
    if label in results.keys():
        results = results[label]

        color = get_hex_colors(palette)
        colors = color * 2

        columns = [f'Clone {idx}' for idx in range(len(aggre.keys()))]
        columns[-1] = 'Clone BG'

        df = pd.DataFrame(data=results[1], columns=results[0], index=columns)
        ax = df.plot(kind='bar', stacked=False, figsize=(40, 10), width=0.9, color=colors)

        plt.xticks(rotation=0, fontsize=25)
        plt.yticks(fontsize=25)
        # plt.title(f'# of progenies produced given an ancester population ({label}) in percentage view', fontsize=28, pad=13)
        plt.title(f'# actual clones for each progeny cluster given an ancester population ({label})', fontsize=28, pad=13)
        plt.legend(bbox_to_anchor=(0.9, -0.1), frameon=False, fontsize=25, ncol=7)
        plt.tick_params(axis='both', which='both', width=2, length=10)

        if logy:
            plt.yscale('log')
            plt.legend(frameon=False, fontsize=22, ncol=1 if len(results[0]) < 10 else 2)

        if save:
            plt.savefig(f'./{save}.svg', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
