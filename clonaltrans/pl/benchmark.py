import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from .gillespie_tree import get_fate_prob, get_hex_colors, find_successive_ones
from .base import get_subplot_dimensions
from matplotlib.lines import Line2D

def min_max(array):
    return (array - min(array)) / (max(array) - min(array))

def generate_consecutive_pairs(items):
    pairs = []
    for i in range(len(items) - 1):
        pairs.append((items[i], items[i + 1]))
    return pairs

def get_cascade_score(
    aggre, 
    clone,
    transit_paths
):
    score = 0

    path_weights = []
    for path in transit_paths:
        consecutive_pairs = generate_consecutive_pairs(path)
        first_pair = consecutive_pairs[0]
        path_weights.append(aggre[clone][first_pair[0]][first_pair[1]])

    path_weights = path_weights / np.sum(path_weights) if np.sum(path_weights) != 0 else path_weights
    
    for idx, path in enumerate(transit_paths):
        consecutive_pairs = generate_consecutive_pairs(path)

        temp_score = 1
        for pair in consecutive_pairs:
            try:
                temp_score *= aggre[clone][pair[0]][pair[1]]
            except:
                pass
        
        score += temp_score * path_weights[idx]

    return score
    
def get_trajectories(
    matrix, 
    start_row, 
    target_col,
    cluster_names
):
    paths = []
    stack = [(start_row, [start_row])]

    while stack:
        node, path = stack.pop()

        if node == target_col:
            # print (f'Transition paths: {list(cluster_names[path])}')
            paths.append(list(cluster_names[path]))
            continue

        for neighbor in range(len(matrix)):
            if matrix[node][neighbor] == 1:
                stack.append((neighbor, path + [neighbor]))

    return paths

def get_transit_path(model, cluster_names, progenitor, fate, selected_fates):
    paga = pd.read_csv(os.path.join(
        model.config['data_loader']['args']['data_dir'], model.config['data_loader']['args']['graphs'],
    ), index_col=0).astype(np.int32)

    transit_paths = get_single_transit_path(cluster_names, progenitor, fate, paga)
    transit_paths_all = get_all_transit_paths(cluster_names, progenitor, selected_fates, paga)

    return transit_paths, transit_paths_all

def get_all_transit_paths(cluster_names, progenitor, selected_fates, paga):
    transit_paths_all = []

    for selected_fate in selected_fates:
        transit_paths_all.extend(
            get_trajectories(
                paga.values, 
                np.where(cluster_names == progenitor)[0][0], 
                np.where(cluster_names == selected_fate)[0][0],
                cluster_names
            )
        )
    
    return transit_paths_all

def get_single_transit_path(cluster_names, progenitor, fate, paga):
    transit_paths = get_trajectories(
        paga.values, 
        np.where(cluster_names == progenitor)[0][0], 
        np.where(cluster_names == fate)[0][0],
        cluster_names
    )
    return transit_paths

def get_tracer_bias(transit_paths, aggre):
    tracer_bias = [-1] * len(aggre.keys())

    for idx, clone in enumerate(aggre.keys()):
        if aggre[clone] != {}:
            tracer_bias[idx] = get_cascade_score(aggre, clone, transit_paths)

    return tracer_bias

def plt_function(
    x,
    y,  
    progenitor, 
    fate, 
    xlabel='GroundTruth',
    ylabel='CloneTracer',
    colors=['blue'],
    legend_elements=[],
    labels=[],
    save=False
):
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    corr, p_value = pearsonr(x, y)
    sns.scatterplot(x=x, y=y, ax=axes, color=colors, s=300)
    axes.plot(
        [x.min(), x.max()], 
        [x.min(), x.max()], 
        linestyle="--", color="grey", zorder=0, linewidth=1
    )
    
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.tick_params(axis='both', labelsize=22)

    axes.set_title(f'Fate bias {progenitor} \u2192 {fate}', fontsize=20)
    axes.set_xlabel(xlabel, fontsize=20)
    axes.set_ylabel(ylabel, fontsize=20)
    # axes.text(0.3, 0.1, f'$Pearson \; Corr = {corr:.3f}$', fontsize=20, transform=axes.transAxes)

    fig.legend(legend_elements, labels, loc='right', fontsize=15, bbox_to_anchor=(1.2, 0.5), frameon=False)

    if save:
        plt.savefig(f'./{save}.svg', dpi=300, bbox_inches='tight', transparent=True)

def get_groundtruth_bias(adata_meta, aggre, transit_paths, transit_paths_all, color):
    perc_trails, legend_elements, labels = [], [], []

    for idx, clone in enumerate(aggre.keys()):
        if aggre[clone] != {}:
            df_temp = adata_meta.obs[adata_meta.obs['meta_clones'] == clone[6:]]
            
            if clone == 'Clone BG':
                df_temp = adata_meta.obs

            max_length = max(len(row) for row in transit_paths_all)
            arr = np.array([row + ['Temp'] * (max_length - len(row)) for row in transit_paths_all])

            denominator = np.unique(np.array(arr).flatten().squeeze())
            denominator = df_temp[df_temp['label_man'].isin(denominator)]
            denominator = len(np.unique(denominator['clones'].values))

            numerator = []
            for path in transit_paths:
                for celltype in path[::-1]:

                    if celltype not in aggre[clone].keys():
                        numerator.append(celltype)
                    else:
                        break
            
            numerator = len(np.unique(df_temp[df_temp['label_man'].isin(np.unique(numerator))]['clones'].values))
            perc_trails.append(numerator / denominator if denominator != 0 else 0)

            legend_elements.append(Line2D([0], [0], marker='o', color=color[idx], markersize=7, linestyle=''))
            labels.append(f'Meta-clone {clone[6:]}')
        else:
            perc_trails.append(-1)
    
    return perc_trails, legend_elements, labels

def compare_gt_pairwise(
    adata_meta,
    progenitor,
    fate,      
    model,
    cluster_names,
    gillespie_dir,
    all_fates,
    init_celltype,
    save=False
):
    color = get_hex_colors('tab20')
    aggre = get_fate_prob(model, cluster_names, gillespie_dir, init_celltype=init_celltype)
    transit_paths, transit_paths_all = get_transit_path(model, cluster_names, progenitor, fate, selected_fates=all_fates)

    tracer_bias = get_tracer_bias(transit_paths, aggre)
    perc_trails, legend_elements, labels = get_groundtruth_bias(adata_meta, aggre, transit_paths, transit_paths_all, color)

    selected = np.where(np.array(tracer_bias) != -1)[0]
    color = np.array(color)[selected]
    tracer_bias = np.array(tracer_bias)[selected]
    perc_trails = np.array(perc_trails)[selected]

    plt_function(perc_trails, tracer_bias, progenitor, fate, 'Percentage of barcode quantity', 'CLADES', color, legend_elements, labels, save)

def compare_gt_all(
    adata_meta,     
    model,
    cluster_names,
    gillespie_dir,
    selected_fates,
    all_fates,
    init_celltype,
    show_fate=True,
    save=False
):
    aggre = get_fate_prob(model, cluster_names, gillespie_dir, init_celltype=init_celltype)
    res_tracer, res_groundtruth = [], []
    scatter_color, legend_elements, labels = [], [], []
    color = get_hex_colors('tab20')

    clones = list(aggre.keys())[0]
    keys = list(aggre[clones].keys())

    paga = pd.read_csv(os.path.join(
        model.config['data_loader']['args']['data_dir'], model.config['data_loader']['args']['graphs'],
    ), index_col=0).astype(np.int32)
    descendents = find_successive_ones(paga, init_celltype[0])
    print (f'Descendents of {init_celltype[0]}: {descendents}')

    for progenitor in keys:
        for fate in list(aggre[clones][progenitor].keys()):
            if fate in selected_fates:
                transit_paths, transit_paths_all = get_transit_path(model, cluster_names, progenitor, fate, selected_fates=all_fates)

                for path in transit_paths:
                    if (init_celltype[0] in path and init_celltype[0] == path[0]) or (init_celltype[0] not in path and path[0] in descendents):
                        tracer_bias = get_tracer_bias(transit_paths, aggre)
                        perc_trails, _, _ = get_groundtruth_bias(adata_meta, aggre, transit_paths, transit_paths_all, color)

                        selected = np.where(np.array(tracer_bias) != -1)[0]
                        tracer_bias = np.array(tracer_bias)[selected]
                        perc_trails = np.array(perc_trails)[selected]

                        res_tracer.append(tracer_bias)
                        res_groundtruth.append(perc_trails)

                        # if show_fate:
                        #     scatter_color.extend([color[np.where(np.array(selected_fates) == fate)[0][0]]] * len(tracer_bias))
                        # else:
                        #     scatter_color.extend([color[np.where(np.array(cluster_names) == progenitor)[0][0]]] * len(tracer_bias))
                        if show_fate:
                            scatter_color.extend([color[np.where(np.array(selected_fates) == fate)[0][0]]])
                        else:
                            scatter_color.extend([color[np.where(np.array(cluster_names) == progenitor)[0][0]]])

    tracer_bias = [np.mean(item) for item in res_tracer]
    perc_trails = [np.mean(item) for item in res_groundtruth]
    # tracer_bias = np.array(res_tracer).flatten()
    # perc_trails = np.array(res_groundtruth).flatten()

    if show_fate:
        for idx, fate in enumerate(selected_fates):
            legend_elements.append(Line2D([0], [0], marker='o', color=color[np.where(np.array(selected_fates) == fate)[0][0]], markersize=7, linestyle=''))
            labels.append(fate)
    
    else:
        remains = [item for item in cluster_names if item not in selected_fates]

        for idx, fate in enumerate(remains):
            legend_elements.append(Line2D([0], [0], marker='o', color=color[np.where(np.array(cluster_names) == fate)[0][0]], markersize=7, linestyle=''))
            labels.append(fate)

    plt_function(np.array(perc_trails), np.array(tracer_bias), 'Progenitors', 'Fates', 'Percentage of barcode quantity', 'CLADES', scatter_color, legend_elements, labels, save)
