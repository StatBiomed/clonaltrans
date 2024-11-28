## Processed data and Model inputs

Here we provide the inputs of the two LT-scSeq datasets used by CLADES, and a pipeline of its generation procedure given the annotated `adata`, denoted as the `prepare_input.ipynb`. 

Specifically, `annotations.csv` `initial_conditions.csv` and `kinetics_array_correction_factor.txt` can be directly generated following the steps given by the notebook, whilst `graph_table.csv` is a refined matrix of PAGA, which requires expert curation of some edges.

To run the model, please place the 4 files within the same directory, and provide the absolute path of that directory to the `main_xxx_xxx.json` in the `./clonaltrans/config/` folder. 