# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.append("..")
from pathlib import Path

# %% [markdown]
# # Attention analysis

# %%
"""
Standalone annotations script for hand pose estimation using POTR HPOES.

! Warning !
The script expects the input data in a .H5 dataset (under "images" key) in a form specified below. For other data
structures, please implement your own logic. Either way, the `depth_maps` should be an array of size (N, 256, 256).
"""

import argparse
import torch
import h5py

import numpy as np
import pandas as pd
from itertools import product

from torch.utils.data import DataLoader

from deformable_potr.models.deformable_potr import DeformablePOTR
from deformable_potr.models.backbone import build_backbone
from deformable_potr.models.deformable_transformer import build_deformable_transformer

from dataset.hpoes_dataset import HPOESOberwegerDataset

from explain.probes import ActivationProbe

from matplotlib import pyplot as plt
import seaborn as sns

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Keypoint and finger connectivity and names

# %% tags=[]
keypoint_names = {0: "little_0",
                  1: "little_1",
                  2: "ring_0",
                  3: "ring_1",
                  4: "middle_0",
                  5: "middle_1",
                  6: "index_0",
                  7: "index_1",
                  8: "thumb_0",
                  9: "thumb_1",
                  10: "thumb_2",
                  11: "wrist_0",
                  12: "wrist_1",
                  13: "palm"
                 }
keypoint_idxs = { "little_0": 0,
                  "little_1": 1,
                  "ring_0": 2,
                  "ring_1": 3,
                  "middle_0": 4,
                  "middle_1": 5,
                  "index_0": 6,
                  "index_1": 7,
                  "thumb_0": 8,
                  "thumb_1": 9,
                   "thumb_2":10,
                   "wrist_0":11,
                   "wrist_1":12,
                   "palm": 13
                 }
keypoint_colors = { "little_0": "r",
                    "little_1": "r",
                    "ring_0": "y",
                    "ring_1": "y",
                    "middle_0": "g",
                    "middle_1": "g",
                    "index_0": "b",
                    "index_1": "b",
                    "thumb_0": "m",
                    "thumb_1": "m",
                    "thumb_2": "m",
                    "wrist_0": "pink",
                    "wrist_1": "pink",
                    "palm": "cyan"
                 }
finger_conns = {"little": [0,1],
              "ring": [2, 3],
              "middle": [4, 5],
              "index": [6,7],
              "thumb": [8, 9, 10],
               "wrist": [11, 12],
               "palm": [13]}
finger_cols = {"little": "r",
              "ring": "y",
              "middle": "g",
              "index": "b",
              "thumb": "m",
               "wrist": "pink",
               "palm": "cyan"}


# %% [markdown]
# ## Running model

# %%
class Args:
    pass

args = Args()
args.weights_file = Path("../.checkpoints/deformable_port/NYU_comrefV2V_3Dproj52_checkpoint_best.pth")
args.input_file = Path("../.data/test_1_comrefV2V_3Dproj.h5")
args.output_file = Path("out_stat.h5")
args.device = "cuda"
args.batch_size = 1

device = torch.device(args.device)

# %%
output_datafile = h5py.File(args.output_file, 'w')

# %%
# Load the input data and checkpoint
print("Loading the input data and checkpoints.")

checkpoint = torch.load(args.weights_file, map_location=device)

# Construct the model from the loaded data
model = DeformablePOTR(
    build_backbone(checkpoint["args"]),
    build_deformable_transformer(checkpoint["args"]),
    num_queries=checkpoint["args"].num_queries,
    num_feature_levels=checkpoint["args"].num_feature_levels
)
model.load_state_dict(checkpoint["model"])
model.eval()
model.to(device)

print("Attaching activation probe")
probe = ActivationProbe(model, activation_recording_mode="output").activation_recording(True)

print("Constructed model successfully.")

# %%
dataset_test = HPOESOberwegerDataset(args.input_file, encoded=True, mode='eval')
data_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, shuffle=False)

# %% [markdown] tags=[]
# ### Specific constants used in deformable attention

# %%
N = 1  # batch size
Len_q = 14  # number of queries, in cross attention module this is number keypoints
n_heads = 8  # attention heads
n_levels = 4  # feature pyramid levels
n_points = 4  # reference points
input_spatial_shapes = torch.tensor([[28, 28], 
                                     [14, 14], 
                                     [7, 7], 
                                     [4, 4]]) # shapes of individual maps from feature pyramid
valid_ratios = torch.tensor([[[1., 1.],[1., 1.], [1., 1.], [1., 1.]]])  # ???

# %%
from collections import namedtuple
AttRow = namedtuple("AttRow", ("sample_id", "layer", "keypoint" ,"head","level", "point", "x", "y", "attention"))
PredRow = namedtuple("PredRow", ("sample_id", "keypoint" ,"px", "py", "pz", "lx", "ly", "lz", "err"))

def create_rows(sample_id, attentions, recorded_sampling_locations):
    rows = []
    for layer_id, layer in enumerate(filter(lambda k: "cross" in k, attentions)):
        rows += [AttRow(sample_id, layer_id, keypoint, head, level, point, 
                     recorded_sampling_locations[layer][-1][0, keypoint, head, level, point, 0].item(),
                     recorded_sampling_locations[layer][-1][0, keypoint, head, level, point, 1].item(),
                     attentions[layer][-1][0,keypoint, head, level, point].item()) 
                 for keypoint, head, level, point in 
                 product(keypoint_names.keys(), range(n_heads), range(n_levels), range(n_points))
                ]
    return rows


# %%
att_rows = []
pred_rows = []
for i, (samples) in enumerate(list(data_loader)[2:]):
    depth_map, label = samples
    print(i)
    depth_map = depth_map.to(device, dtype=torch.float32)

    results = model(depth_map).detach().cpu().numpy()

    attentions = probe.attentions  
    recorded_sampling_locations = probe.sampling_locations
    
    att_rows += create_rows(i, attentions, recorded_sampling_locations)
    pred_rows += [PredRow(i, keypoint, *results[0, keypoint], 
                                       *label.cpu().numpy()[0, keypoint],
                                      np.linalg.norm(results[0, keypoint] - label.cpu().numpy()[0, keypoint])
                         ) for keypoint in keypoint_names.keys()]
    
    break
att_df = pd.DataFrame(att_rows)
pred_df = pd.DataFrame(pred_rows)

# %%
decoder_reference_points = probe.output_activations["transformer-reference_points"].cpu()
decoder_reference_points = decoder_reference_points.sigmoid()
decoder_reference_points = decoder_reference_points[:, :, None] * valid_ratios[:, None]
decoder_reference_points = decoder_reference_points.detach().cpu().numpy()

# %% [markdown]
# ## The analysis

# %%
pred_df

# %%
att_df["dist"] = np.linalg.norm(att_df[["x", "y"]] - np.array([0.5, 0.5]), axis=1)

# %%
att_df = att_df.merge(pred_df, on="keypoint")

# %%
att_df.plot(kind="scatter", x="dist", y="attention")

# %%
sns.scatterplot(data=att_df, x="dist", y="attention", hue="layer")

# %%
plt.figure(figsize=(20, 15))
sns.scatterplot(data=att_df, x="x", y="y", hue="layer", size="attention")

# %%
sns.scatterplot(data=att_df, x="dist", y="attention", hue="level")

# %%
sns.scatterplot(data=att_df, x="dist", y="attention", hue="err")

# %%
df = att_df
df["keypoint"] = df["keypoint"].map(keypoint_names)
for l in range(6):
    fig = plt.figure(figsize=(25, 6))    
    ax1, ax2, ax3, ax4 = fig.subplots(1, 4)
    ax1.set_title(f"Layer {l}: attention vs dist")
    sns.scatterplot(data=df[df["layer"] == l], x="dist", y="attention", ax=ax1)
    ax2.set_title(f"sampling points distribution vs attenion")
    sns.scatterplot(data=df[df["layer"] == l], x="x", y="y", hue="attention", ax=ax2)
    ax2.plot([0, 0, 1,1, 0], [0, 1, 1, 0, 0], color="red")
    ax3.set_title(f"sampling points distribution vs level")
    sns.scatterplot(data=df[df["layer"] == l], x="x", y="y", hue="level", ax=ax3)
    ax3.plot([0, 0, 1,1, 0], [0, 1, 1, 0, 0], color="red")
    sns.scatterplot(data=df[df["layer"] == l], x="x", y="y", hue="keypoint", palette="tab20", ax=ax4)
    ax4.plot([0, 0, 1,1, 0], [0, 1, 1, 0, 0], color="red")

# %%
df = att_df[att_df["keypoint"] == keypoint_idxs["little_0"]]
for l in range(6):
    fig = plt.figure(figsize=(20, 6))    
    ax1, ax2, ax3 = fig.subplots(1, 3)
    ax1.set_title(f"Layer {l}: attention vs dist")
    sns.scatterplot(data=df[df["layer"] == l], x="dist", y="attention", ax=ax1)
    ax2.set_title(f"sampling points distribution vs attenion")
    sns.scatterplot(data=df[df["layer"] == l], x="x", y="y", hue="attention", ax=ax2)
    ax2.plot([0, 0, 1,1, 0], [0, 1, 1, 0, 0], color="red")
    ax3.set_title(f"sampling points distribution vs level")
    sns.scatterplot(data=df[df["layer"] == l], x="x", y="y", hue="level", ax=ax3)
    ax3.plot([0, 0, 1,1, 0], [0, 1, 1, 0, 0], color="red")

# %%
