import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from .gillespie_tree import get_fate_prob, get_hex_colors
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
    for path in transit_paths:
        consecutive_pairs = generate_consecutive_pairs(path)

        temp_score = 1
        for pair in consecutive_pairs:
            try:
                temp_score *= aggre[clone][pair[0]][pair[1]]
            except:
                pass
        
        score += temp_score
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

def get_cospar_bias(adata_cospar, adata_meta, progenitor, fate):
    adata_cospar.obs['meta_clones'] = adata_meta.obs['meta_clones'].values
    adata_cospar.obs['Clone_ID'] = adata_meta.obs['Clone_ID'].values

    df = adata_cospar.obs[adata_cospar.obs[f'fate_map_transition_map_{fate}'] >= 0]
    df = df[df['state_info'] == progenitor]

    cospar_bias = pd.DataFrame(
            df.groupby('meta_clones')[f'fate_map_transition_map_{fate}'].mean()
        )
    cospar_bias = cospar_bias.fillna(-1)
    cospar_bias = cospar_bias[f'fate_map_transition_map_{fate}'].values
    return cospar_bias

def get_transit_path(model, cluster_names, progenitor, fate):
    paga = pd.read_csv(os.path.join(
        model.config['data_loader']['args']['data_dir'], model.config['data_loader']['args']['graphs'],
    ), index_col=0).astype(np.int32)

    transit_paths = get_trajectories(
        paga.values, 
        np.where(cluster_names == progenitor)[0][0], 
        np.where(cluster_names == fate)[0][0],
        cluster_names
    )

    selected_fates = ["Ery", "Meg", "Eos", "Mast", "DC", "Mono", "Neu", "pDC", "Ly", 'Baso']
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

    return transit_paths, transit_paths_all

def get_tracer_bias(transit_paths, aggre):
    tracer_bias = []
    for clone in aggre.keys():
        if int(clone[6:]) + 1 != len(aggre.keys()):
            tracer_bias.append(get_cascade_score(aggre, clone, transit_paths))
    
    return tracer_bias

def plt_function(
    tracer_bias, 
    cospar_bias, 
    progenitor, 
    fate, 
    axes,
    xlabel='CloneTracer',
    ylabel='CoSpar',
    colors=['blue']
):
    corr, p_value = pearsonr(tracer_bias, cospar_bias)
    sns.scatterplot(x=tracer_bias, y=cospar_bias, ax=axes, color=colors, s=100)
    axes.plot(
        [tracer_bias.min(), tracer_bias.max()], 
        [tracer_bias.min(), tracer_bias.max()], 
        linestyle="--", color="grey", zorder=0
    )
    
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.tick_params(axis='both', labelsize=17)

    axes.set_title(f'Fate bias {progenitor} \u2192 {fate}', fontsize=18)
    axes.set_xlabel(xlabel, fontsize=18)
    axes.set_ylabel(ylabel, fontsize=18)
    axes.text(0.3, 0.1, f'$Pearson \; Corr = {corr:.3f}$', fontsize=18, transform=axes.transAxes)

def get_groundtruth_bias(adata_meta, aggre, transit_paths, transit_paths_all, color):
    perc_trails, legend_elements, labels = [], [], []
    for idx, clone in enumerate(aggre.keys()):
        if int(clone[6:]) + 1 != len(aggre.keys()):
            df_temp = adata_meta.obs[adata_meta.obs['meta_clones'] == clone[6:]]

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
    
    return perc_trails, legend_elements, labels

def with_cospar(
    adata_cospar,
    adata_meta,
    progenitor,
    fate,      
    model,
    cluster_names,
    gillespie_dir,
    save=False
):
    aggre = get_fate_prob(model, cluster_names, gillespie_dir)
    transit_paths, transit_paths_all = get_transit_path(model, cluster_names, progenitor, fate)

    cospar_bias = get_cospar_bias(adata_cospar, adata_meta, progenitor, fate)
    tracer_bias = get_tracer_bias(transit_paths, aggre)

    color = get_hex_colors('tab20')
    perc_trails, legend_elements, labels = get_groundtruth_bias(adata_meta, aggre, transit_paths, transit_paths_all, color)

    color = np.array(color)[np.where(cospar_bias != -1)[0]]
    tracer_bias = np.array(tracer_bias)[np.where(cospar_bias != -1)[0]]
    perc_trails = np.array(perc_trails)[np.where(cospar_bias != -1)[0]]
    cospar_bias = np.array(cospar_bias)[np.where(cospar_bias != -1)[0]]

    cospar_bias = min_max(cospar_bias)
    tracer_bias = min_max(tracer_bias)
    perc_trails = min_max(perc_trails)

    rows, cols, figsize = get_subplot_dimensions(3, fig_height_per_row=4)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] + 5, figsize[1]))

    plt_function(tracer_bias, cospar_bias, progenitor, fate, axes[0], 'CLADES', 'CoSpar', color)
    plt_function(perc_trails, cospar_bias, progenitor, fate, axes[1], 'Ground Truth', 'CoSpar', color)
    plt_function(perc_trails, tracer_bias, progenitor, fate, axes[2], 'Ground Truth', 'CLADES', color)

    fig.legend(legend_elements, labels, loc='right', fontsize=18, bbox_to_anchor=(1.02, 0.5), frameon=False)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=True)

def with_cospar_all(
    adata_cospar,
    adata_meta,     
    model,
    cluster_names,
    gillespie_dir,
    selected_fates,
    show_fate=True,
    save=False
):
    aggre = get_fate_prob(model, cluster_names, gillespie_dir)
    res_cospar, res_tracer, res_groundtruth = [], [], []
    scatter_color, legend_elements, labels = [], [], []
    color = get_hex_colors('tab20')

    clones = list(aggre.keys())[0]
    keys = list(aggre[clones].keys())

    for progenitor in keys:
        for fate in list(aggre[clones][progenitor].keys()):
            if fate in selected_fates:
                transit_paths, transit_paths_all = get_transit_path(model, cluster_names, progenitor, fate)

                cospar_bias = get_cospar_bias(adata_cospar, adata_meta, progenitor, fate)
                tracer_bias = get_tracer_bias(transit_paths, aggre)
                perc_trails, _, _ = get_groundtruth_bias(adata_meta, aggre, transit_paths, transit_paths_all, color)

                tracer_bias = np.array(tracer_bias)[np.where(cospar_bias != -1)[0]]
                perc_trails = np.array(perc_trails)[np.where(cospar_bias != -1)[0]]
                cospar_bias = np.array(cospar_bias)[np.where(cospar_bias != -1)[0]]

                res_cospar.append(cospar_bias)
                res_tracer.append(tracer_bias)
                res_groundtruth.append(perc_trails)

                if show_fate:
                    scatter_color.append(color[np.where(np.array(selected_fates) == fate)[0][0]])
                else:
                    scatter_color.append(color[np.where(np.array(cluster_names) == progenitor)[0][0]])

    cospar_bias = [min_max(item) for item in res_cospar]
    tracer_bias = [min_max(item) for item in res_tracer]
    perc_trails = [min_max(np.array(item)) for item in res_groundtruth]

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    tracer_bias = [pearsonr(tracer_bias[idx], perc_trails[idx])[0] for idx, item in enumerate(tracer_bias)]
    cospar_bias = [pearsonr(cospar_bias[idx], perc_trails[idx])[0] for idx, item in enumerate(cospar_bias)]
    plt_function(np.array(tracer_bias), np.array(cospar_bias), 'Progenitors', 'Fates', axes, 'CLADES / Ground Truth', 'CoSpar / Ground Truth', scatter_color)

    if show_fate:
        for idx, fate in enumerate(selected_fates):
            legend_elements.append(Line2D([0], [0], marker='o', color=color[np.where(np.array(selected_fates) == fate)[0][0]], markersize=7, linestyle=''))
            labels.append(fate)
        fig.legend(legend_elements, labels, loc='right', fontsize=18, bbox_to_anchor=(1.3, 0.5), frameon=False)
    
    else:
        remains = [item for item in cluster_names if item not in selected_fates]

        for idx, fate in enumerate(remains):
            legend_elements.append(Line2D([0], [0], marker='o', color=color[np.where(np.array(cluster_names) == fate)[0][0]], markersize=7, linestyle=''))
            labels.append(fate)
        fig.legend(legend_elements, labels, loc='right', fontsize=18, bbox_to_anchor=(2, 0.5), frameon=False)

    if save:
        plt.savefig(f'./{save}.svg', dpi=600, bbox_inches='tight', transparent=True)