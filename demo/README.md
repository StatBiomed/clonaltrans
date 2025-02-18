## Processed data and Model inputs

Here we provide the inputs of the two LT-scSeq datasets used by CLADES, and a pipeline of its generation procedure given the annotated `adata`, denoted as the `prepare_input.ipynb`. 

Specifically, `annotations.csv` and `kinetics_corrected.txt` can be directly generated following the steps given by the notebook, whilst `paga_transitions.csv` is a refined matrix of PAGA, which requires expert curation on certain edges.

To run the model, please place the 3 files within the same directory, and provide the absolute path of that directory to the configuration file `main_xxx_xxx.json` in the `./clonaltrans/config/` folder. 