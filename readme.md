# FLAT
## Tabular Few-Shot Generalization Across Heterogeneous Feature Spaces

### How to run pre-trained models

Included in `saves` folder are 3 FLAT models pretrained on all 118 UCI datasets (without excluding any datasets for the testing split $\mathcal{D}_{test}$). These are intended to be our most general models, which users can use on their own datasets. 
A minimal example of how to load the pre-trained models is presented in `Fewshot/load_pretrained_FLAT_example.ipynb`
These models are for binary classification. Models are named after how many meta-rows were present during training, e.g. `saves/model_10_meta/model.pt` is trained on $N^{meta}=10$. Using the model with the most similar training $N^{meta}$ as test $N^{meta}$ should generally give the best results. 

### How to train FLAT models from scratch

1. The UCI datasets used for our experiments can be found in `datasets/data`

2. To train a FLAT model, set the configuration file in `Fewshot/configs.toml`

    Most of the settings here are pre-selected model hyperparamters. 
    To train the model on the medical datasets set (Section 4.1):
    ```
    fold_no={fold_no}
    folds_file="med_folds"
    ```

    To train the model on all UCI datasets set (Section 4.2):
    ```
    fold_no={fold_no}
    folds_file="folds"
    ```

    `{fold_no}` should be an integer between 0 to $N-1$, which indicates which of the $N$-folds within the dataset collection is held-out for testing.
    For the medical example, $N=10$. For all datasets $N=4$.

3. To train the FLAT model, run `python Fewshot/train.py`. Models are automatically saved in `saves/{run_no}`.