import sys
sys.path.append('./Fewshot/')
from Fewshot.load_pretrained import FLAT, FLATadapt
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import torch

# load a custom dataset
data = load_breast_cancer()
X, y = data.data, data.target
df = pd.DataFrame(X)
df['y'] = y

num_meta = 10
num_target = 10

# initialize FLAT models
model_file_path = 'saves/model_10_meta/model.pt'
config_file_path = 'saves/model_10_meta/configs.toml'

flat = FLAT(model_file_path=model_file_path, config_file_path=config_file_path)
flatadapt = FLATadapt(model_file_path=model_file_path, config_file_path=config_file_path)

# sample meta and target data
subsample_df = df.sample(num_meta + num_target)

# standardize features xs
subsample_xs = subsample_df.drop('y', axis=1).values
subsample_xs = (subsample_xs - subsample_xs.mean(axis=0)) / subsample_xs.std(axis=0)

xs_meta, xs_target = subsample_xs[:num_meta], subsample_xs[num_meta:]
ys_meta, ys_target = subsample_df['y'].values[:num_meta], subsample_df['y'].values[num_meta:]

# Fit FLAT to the meta data D_meta = (xs_meta, ys_meta)
flat.fit(
    xs_meta=torch.tensor(xs_meta, dtype=torch.float32),
    ys_meta=torch.tensor(ys_meta, dtype=torch.long)
)
flatadapt.fit(
    xs_meta=torch.tensor(xs_meta, dtype=torch.float32),
    ys_meta=torch.tensor(ys_meta, dtype=torch.long)
)

# Predict on the target features xs_target
ys_pred_target_flat = flat.predict(
    xs_target=torch.tensor(xs_target, dtype=torch.float32)
)
ys_pred_target_flatadapt = flatadapt.predict(
    xs_target=torch.tensor(xs_target, dtype=torch.float32)
)

# Print results and compare against ys_target
print("======FLAT========")
print("Predicted labels: ", ys_pred_target_flat)
print("True labels: ", ys_target)
print("Accuracy: ", np.mean(ys_pred_target_flat == ys_target))
print("\n")
print("======FLATadapt========")
print("Predicted labels: ", ys_pred_target_flatadapt)
print("True labels: ", ys_target)
print("Accuracy: ", np.mean(ys_pred_target_flatadapt == ys_target))