import numpy as np
import pandas as pd
from natsort import natsorted
from itertools import product
import scanpy as sc
import os
from scipy.sparse import load_npz
import shutil
import os

def prepare_manuscript_data(config, logger, dataset='cordblood'):
    logger.info('Estimating clonal specific transition rates.\n')
    config['data_loader']['args']['logger'] = logger

    config['nni_data']['data_dir'] = os.path.join(config._log_dir, 'nni_data')
    os.makedirs(config['nni_data']['data_dir'], exist_ok=True)
    logger.info(f"Data directory for NNI is created at {config['nni_data']['data_dir']}\n")

    if dataset == 'cordblood':
        prepare_cordblood_input(config, logger)
    if dataset == 'weinreb':
        prepare_weinreb_input(config, logger)
    else:
        raise ValueError('Invalid dataset type. Please select either cordblood or weinreb.\n')

    shutil.copy(
        os.path.join(config['data_loader']['args']['data_dir'], config['data_loader']['args']['graphs']), 
        os.path.join(config['nni_data']['data_dir'], config['data_loader']['args']['graphs'])
    )
    logger.info(f"Graph file is copied to the data directory {os.path.join(config['nni_data']['data_dir'], config['data_loader']['args']['graphs'])}")

    config['data_loader']['args']['data_dir'] = config['nni_data']['data_dir']
    logger.info(f"Data directory is set to {config['data_loader']['args']['data_dir']}\n")

    return config

def prepare_cordblood_input(config, logger):
    adata = sc.read_h5ad(config['nni_data']['model_path'])
    clone_data = pd.read_pickle(config['nni_data']['clone_data_path'])

    adata.obs['def_lab'] = adata.obs['def_lab'].astype('str')
    adata.obs.loc[adata.obs['leiden'] == '7', 'def_lab'] = 'HSC/MPP 2'
    adata.obs.loc[adata.obs['def_lab'] == 'HSC_MPP', 'def_lab'] = 'HSC/MPP 1'
    adata.obs.loc[adata.obs['def_lab'] == 'DC precursor ', 'def_lab'] = 'DC precursor'

    clones = clone_data[clone_data.columns[14:]]
    clones = clones.fillna(0)
    clones = clones.loc[adata.obs_names].copy()

    adata.obs['Timepoint'] = adata.obs['Timepoint'].astype('str')
    adata.obs['time_label_full'] = adata.obs['Timepoint'] + '_' + adata.obs['def_lab']
    adata.obs['clones'] = ['Clone_'  + str(np.where(row != 0)[0] + 1)[1:-1] for idx, row in clones.iterrows()]

    rename_cell = adata.obs[adata.obs['time_label_full'] == 'Day17_HSC/MPP 1']

    adata.obs.loc[rename_cell.index, 'def_lab'] = rename_cell['transferred_labels'].values
    adata.obs.loc[adata.obs['def_lab'] == 'DC2', 'def_lab'] = 'DC'
    adata.obs['time_label_full'] = adata.obs['Timepoint'] + '_' + adata.obs['def_lab']

    adata_pos = adata[adata.obs['GFP'] == 'GFP+', :]
    adata_pos_barcode = adata_pos[[item != 'Clone_' for item in list(adata_pos.obs['clones'])], :]



    mat = clones.loc[adata_pos_barcode.obs.index.intersection(clones.index)]
    mat['time_label_full'] = adata_pos_barcode.obs['time_label_full']
    mat = mat.groupby('time_label_full').sum().T
    mat.columns.name = ''
    logger.info(f"Clone feature matrix is created with shape {mat.shape}")

    mat = sc.AnnData(mat)
    sc.pp.filter_cells(mat, min_counts=3)
    sc.pp.filter_genes(mat, min_counts=1)
    logger.info(f"Clone feature matrix is filtered with shape {mat.shape}")
    
    sc.tl.pca(mat)
    sc.pp.neighbors(mat, n_neighbors=15)
    sc.tl.umap(mat)
    sc.tl.leiden(mat, resolution=config['nni_data']['leiden_resolution'])
    logger.info(f"Leiden clustering is performed with resolution {config['nni_data']['leiden_resolution']}\n")

    df_leiden = pd.DataFrame(mat.obs.leiden.value_counts())
    df_clones = pd.DataFrame({'count': [len(set(adata.obs['clones']))]})
    df_leiden = pd.concat([df_leiden, df_clones], ignore_index=True)
    df_leiden.columns = ['leiden']
    df_leiden.to_csv(os.path.join(config['nni_data']['data_dir'], 'initial_condition.csv'))
    logger.info(f"Number of meta-clones: {len(df_leiden) - 1}")
    logger.info(f"Initial condition is saved to {os.path.join(config['nni_data']['data_dir'], 'initial_condition.csv')}\n")



    del adata.uns['def_lab_colors']
    adata_clones_filter = adata_pos_barcode[[i in list(mat.obs.index) for i in adata_pos_barcode.obs['clones'].values], :]

    leiden = []
    for clone in adata_clones_filter.obs['clones'].values:
        leiden.append(mat.obs.loc[clone]['leiden'])
    adata_clones_filter.obs['Meta clones'] = leiden

    adata.obs['Meta clones'] = np.zeros(adata.shape[0]) - 1
    for i in list(adata_clones_filter.obs.index):
        adata.obs.at[i, 'Meta clones'] = adata_clones_filter.obs.loc[i]['Meta clones']
    adata.obs['Meta clones'] = [str(i) for i in adata.obs['Meta clones']]

    group_labels = [str(i) for i in list(df_leiden.index)[:-1]]
    adata.obs['group'] = pd.Categorical(adata.obs['Meta clones'], categories=group_labels)

    sc.tl.paga(adata, groups='def_lab')



    times = config['nni_data']['time_available']
    size_leiden = len(set(mat.obs.leiden))
    size_time = len(times)
    size_pops = len(set(adata_clones_filter.obs.def_lab))
    logger.info(f"Dimension of kinetics array (time, meta-clone, population): {size_time} x {size_leiden} x {size_pops}")

    leiden = natsorted(list(set(mat.obs.leiden))) 

    df = pd.DataFrame(data=mat.X, columns=list(mat.var.index), index=list(mat.obs.index))
    df = df.astype('float32')
    df.loc[:, df.columns.str.startswith('Day3_')] = df.loc[:, df.columns.str.startswith('Day3_')] * config['nni_data']['scaling_factor'][0]
    df.loc[:, df.columns.str.startswith('Day10_')] = df.loc[:, df.columns.str.startswith('Day10_')] * config['nni_data']['scaling_factor'][1]
    df.loc[:, df.columns.str.startswith('Day17_')] = df.loc[:, df.columns.str.startswith('Day17_')] * config['nni_data']['scaling_factor'][2]
    df = df.round(0)
    logger.info(f"Scaling factors are applied {config['nni_data']['scaling_factor']}")

    pops = [
        'HSC/MPP 1', 'HSC/MPP 2', 'MEMP', 'Mast cell', 
        'Early Erythroid', 'Mid Erythroid', 'Late Erythroid', 
        'NMP', 'Mono precur', 'Monocyte', 
        'DC precursor', 'DC'
    ]

    kinetics = np.zeros((size_leiden + 1, size_time, size_pops), dtype=np.float32)

    for (idi, leid), (idj, time), (idk, pop) in product(enumerate(leiden), enumerate(times), enumerate(pops)):
        if 'Day' + str(time) + '_' + pop in mat.var_names:
            kinetics[idi, idj, idk] = np.array(df[mat.obs.leiden == leid]['Day' + str(time) + '_' + pop].values).sum()

    for (idj, time), (idk, pop) in product(enumerate(times), enumerate(pops)):
        if 'Day' + str(time) + '_' + pop in adata.obs['time_label_full'].values:
            num = adata[adata.obs['time_label_full'] == 'Day' + str(time) + '_' + pop, :].shape[0]
            if str(time) == '3':
                num *= config['nni_data']['scaling_factor'][0]
            if str(time) == '10':
                num *= config['nni_data']['scaling_factor'][1]
            if str(time) == '17':
                num *= config['nni_data']['scaling_factor'][2]
            kinetics[-1, idj, idk] = int(num)
    
    reshaped_kinetics = np.reshape(kinetics, (size_leiden + 1, size_time * size_pops))
    np.savetxt(os.path.join(config['nni_data']['data_dir'], 'kinetics_array_correction_factor.txt'), reshaped_kinetics, fmt='%.0f')
    logger.info(f"Updated dimension of kinetics array (size_leiden + 1, size_time * size_pops): {reshaped_kinetics.shape}")
    logger.info(f"Kinetics array is saved to {os.path.join(config['nni_data']['data_dir'], 'kinetics_array_correction_factor.txt')}")

    clones = ['Meta-clone ' + str(i) for i in list(df_leiden.index)[:-1]]
    clones.append('BG')

    annots = pd.DataFrame(data=[clones, pops], index=['clones', 'populations']).T
    annots.to_csv(os.path.join(config['nni_data']['data_dir'], 'annotations.csv'), index=False)
    logger.info(f"Len of meta-clones: {len(clones) - 1}, Len of populations: {len(pops)}")
    logger.info(f"Annotations are saved to {os.path.join(config['nni_data']['data_dir'], 'annotations.csv')}\n")

def prepare_weinreb_input(config, logger):
    adata = sc.read_h5ad(config['nni_data']['model_path'])
    clone_data = load_npz(config['nni_data']['clone_data_path'])

    adata.obs['Time_point'] = adata.obs['Time_point'].astype('category')
    clones_used = clone_data.todense()[np.array([int(item) for item in adata.obs.index])]

    index, ids = np.where(clones_used != 0)
    clones_label = []

    for idx in range(adata.shape[0]):
        clones_label.append(f'Clone_{ids[np.where(index == idx)[0][0]]}' if idx in index else 'Clone_nan')
    
    adata.obs['clones'] = clones_label

    adata_clone = adata.obs[adata.obs['clones'] != 'Clone_nan']
    adata_clone['pop_time'] = adata_clone['label_man'].astype(str) + '_' + adata_clone['Time_point'].astype(str)
    adata_clone = adata_clone.drop(labels=['time_cat', 'leiden', 'comb'], axis=1)
    
    df = adata_clone.groupby(by=['pop_time', 'clones']).size().reset_index(name='counts')
    df = df.sort_values(by=['pop_time'])

    df = df.pivot(index='clones', columns='pop_time', values='counts')
    df.index.name = None
    df.columns.name = None
    df = df.fillna(0.0)

    adata.obs['Time_point'] = adata.obs['Time_point'].astype(str)

    mat = sc.AnnData(df, dtype=np.float32)
    df.columns = mat.var.index
    logger.info(f"Clone feature matrix is created with shape {mat.shape}")

    sc.pp.filter_cells(mat, min_counts=3)
    sc.pp.filter_genes(mat, min_counts=1)
    logger.info(f"Clone feature matrix is filtered with shape {mat.shape}")

    sc.tl.pca(mat)
    sc.pp.neighbors(mat, n_neighbors=10, n_pcs=50)
    sc.tl.umap(mat)
    sc.tl.leiden(mat, resolution=config['nni_data']['leiden_resolution'])
    logger.info(f"Leiden clustering is performed with resolution {config['nni_data']['leiden_resolution']}\n")

    df_leiden = pd.DataFrame(mat.obs.leiden.value_counts())
    df_clones = pd.DataFrame({'count': [len(set(adata.obs['clones']))]})
    df_leiden = pd.concat([df_leiden, df_clones], ignore_index=True)
    df_leiden.columns = ['leiden']
    df_leiden.to_csv(os.path.join(config['nni_data']['data_dir'], 'initial_condition.csv'))
    logger.info(f"Number of meta-clones: {len(df_leiden) - 1}")
    logger.info(f"Initial condition is saved to {os.path.join(config['nni_data']['data_dir'], 'initial_condition.csv')}\n")

    adata_clones_filter = adata[adata.obs['clones'] != 'Clone_nan', :]
    adata_clones_filter = adata_clones_filter[adata_clones_filter.obs['clones'].isin(mat.obs.index), :]

    leiden = []
    for clone in adata_clones_filter.obs['clones'].values:
        leiden.append(mat.obs.loc[clone]['leiden'])
    adata_clones_filter.obs['Meta clones'] = leiden

    adata.obs['Meta clones'] = np.zeros(adata.shape[0]) - 1
    for i in list(adata_clones_filter.obs.index):
        adata.obs.at[i, 'Meta clones'] = adata_clones_filter.obs.loc[i]['Meta clones']
    adata.obs['Meta clones'] = [str(i) for i in adata.obs['Meta clones']]

    group_labels = [str(i) for i in list(df_leiden.index)[:-1]]
    adata.obs['group'] = pd.Categorical(adata.obs['Meta clones'], categories=group_labels)

    sc.tl.paga(adata, groups='label_man')

    times = config['nni_data']['time_available']
    size_leiden = len(set(mat.obs.leiden))
    size_time = len(times)
    size_pops = len(set(adata_clones_filter.obs.label_man))
    logger.info(f"Dimension of kinetics array (time, meta-clone, population): {size_time} x {size_leiden} x {size_pops}")

    leiden = natsorted(list(set(mat.obs.leiden))) 

    scaling_factors = config['nni_data']['scaling_factor']
    logger.info(f"Scaling factors are applied {config['nni_data']['scaling_factor']}")

    graph = pd.read_csv(os.path.join(config['data_loader']['args']['data_dir'], config['data_loader']['args']['graphs']), index_col=0)
    graph = graph.fillna(0.0)

    kinetics = np.zeros((size_leiden + 1, size_time, size_pops), dtype=np.float32)
    pops = graph.columns

    for (idi, leid), (idj, time), (idk, pop) in product(enumerate(leiden), enumerate(times), enumerate(pops)):
        kinetics[idi, idj, idk] = int(adata_clones_filter.obs[(adata_clones_filter.obs['Time_point'] == time) & (adata_clones_filter.obs['Meta clones'] == leid) & (adata_clones_filter.obs['label_man'] == pop)].shape[0] * scaling_factors[idj])

    for (idj, time), (idk, pop) in product(enumerate(times), enumerate(pops)):
        kinetics[-1, idj, idk] = int(adata.obs[(adata.obs['Time_point'] == time) & (adata.obs['label_man'] == pop)].shape[0] * scaling_factors[idj])
    
    reshaped_kinetics = np.reshape(kinetics, (size_leiden + 1, size_time * size_pops))
    np.savetxt(os.path.join(config['nni_data']['data_dir'], 'kinetics_array_correction_factor.txt'), reshaped_kinetics, fmt='%.0f')
    logger.info(f"Updated dimension of kinetics array (size_leiden + 1, size_time * size_pops): {reshaped_kinetics.shape}")
    logger.info(f"Kinetics array is saved to {os.path.join(config['nni_data']['data_dir'], 'kinetics_array_correction_factor.txt')}")

    clones = ['Meta-clone ' + str(i) for i in list(df_leiden.index)[:-1]]
    clones.append('BG')

    annots = pd.DataFrame(data=[clones, pops], index=['clones', 'populations']).T
    annots.to_csv(os.path.join(config['nni_data']['data_dir'], 'annotations.csv'), index=False)
    logger.info(f"Len of meta-clones: {len(clones) - 1}, Len of populations: {len(pops)}")
    logger.info(f"Annotations are saved to {os.path.join(config['nni_data']['data_dir'], 'annotations.csv')}\n")