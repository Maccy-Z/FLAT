Pretrained FLAT models. 

Included here are FLAT models pretrained on all 118 UCI datasets. These are intended to be our best models, which users can use on their own datasets. 
The config.toml file is required to load the models. Paste them with the chosen model together into a directory, then load the model using comparison.py. 

These models are for binary classification. Models are named after how many meta-rows were present during training, e.g. model-10.pt is trained on N_meta=10. Using the model with the most similar train N_meta as test N_meta should generally give the best results. 
