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
# # Attention vizualization

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
from tqdm import tqdm

from torch.utils.data import DataLoader

from deformable_potr.models.deformable_potr import DeformablePOTR
from deformable_potr.models.backbone import build_backbone
from deformable_potr.models.deformable_transformer import build_deformable_transformer

from deformable_potr.util.misc import make_path

from dataset.hpoes_dataset import HPOESOberwegerDataset

from explain.probes import ActivationProbe

from matplotlib import pyplot as plt

# %% [markdown] tags=[]
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


# %% tags=[]
def plot_skeleton(joints, style, linewidth = 2.0):
    for fid in finger_conns.keys():
        finger = joints[finger_conns[fid]]
        plt.plot(finger[:, 0], finger[:, 1], linestyle=style, color=finger_cols[fid], linewidth = linewidth)

def plot_points(joints, style):
    plt.scatter(joints[:, 0], joints[:, 1], facecolors=list(keypoint_colors.values()), marker=style)


# %%
def plot_attentions(sample_id, img_data, results, labels, layer, sampling_locations, attention_weights):
    results = ((results + 1) / 2) * 224
    labels = ((labels + 1) / 2) * 224
    
    for keypoint, keypoint_idx in keypoint_idxs.items():
        plt.imshow(img_data)
        plot_points(results, "o")
        plot_skeleton(results, "-")
        plot_points(labels, ".")
        plt.plot(results[keypoint_idx, 0], results[keypoint_idx, 1], marker=".")
        for head in range(8):
            plt.scatter(224 * sampling_locations[keypoint_idx, head, :,:, 0].flatten(), 
                        224 * sampling_locations[keypoint_idx, head, :,:, 1].flatten(), 
                        sizes=50*attention_weights[keypoint_idx, head, :].flatten(),
                        marker="o", facecolors='None', edgecolors=keypoint_colors[keypoint])
        plt.savefig(make_path("output", str(sample_id), f"{layer}_{keypoint}.png"), bbox_inches="tight", dpi=400)
        plt.close()


# %% jupyter={"source_hidden": true} tags=[]
class Args:
    pass

args = Args()
args.weights_file = Path("../.checkpoints/deformable_port/NYU_comrefV2V_3Dproj52_checkpoint_best.pth")
args.input_file = Path("../.data/test_1_comrefV2V_3Dproj.h5")
args.output_file = Path("out_vism.h5")
args.device = "cuda"
args.batch_size = 1

device = torch.device(args.device)

# %% jupyter={"source_hidden": true} tags=[]
output_datafile = h5py.File(args.output_file, 'w')

# %% jupyter={"source_hidden": true} tags=[]
# Load the input data and checkpoint
print("Loading the input data and checkpoints.")

checkpoint = torch.load(args.weights_file, map_location=device)

output_list = []

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

# %% tags=[]
dataset_test = HPOESOberwegerDataset(args.input_file, encoded=True, mode='eval')
data_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, shuffle=False)

# %% [markdown] tags=[]
# ### Specific constants used in deformable attention

# %% tags=[]
N = 1  # batch size
Len_q = 14  # number of queries, in cross attention module this is number of keypoints
n_heads = 8  # attention heads
n_levels = 4  # feature pyramid levels
n_points = 4  # reference points
input_spatial_shapes = torch.tensor([[28, 28], 
                                     [14, 14], 
                                     [7, 7], 
                                     [4, 4]]) # shapes of individual maps from feature pyramid
valid_ratios = torch.tensor([[[1., 1.],[1., 1.], [1., 1.], [1., 1.]]])  # ???

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Reference points
# Reference points to which `sampling_offsets` are added

# %%
decoder_reference_points = probe.output_activations["transformer-reference_points"].cpu()
decoder_reference_points = decoder_reference_points.sigmoid()
decoder_reference_points = decoder_reference_points[:, :, None] * valid_ratios[:, None]
decoder_reference_points.shape

# %% [markdown] tags=[]
# ### Offset normalizer
# Offsets are normalized by feature map resolution.

# %% jupyter={"source_hidden": true, "outputs_hidden": true} tags=[]
offset_normalizer = torch.stack([input_spatial_shapes[..., 1], 
                                 input_spatial_shapes[..., 0]], -1)
offset_normalizer

# %% tags=[]
for i, (samples) in enumerate(list(data_loader)):
    depth_map, label = samples
    print(i)
    depth_map = depth_map.to(device, dtype=torch.float32)

    results = model(depth_map).detach().cpu().numpy()

    attentions = probe.attentions
    recorded_sampling_locations = probe.sampling_locations
    
    img_data = depth_map.cpu().numpy().transpose((2, 3, 1, 0))[..., 0]
    img_data = (img_data - img_data.min()) / np.abs(img_data.max() - img_data.min())
    
    for layer_id, layer in tqdm(enumerate(filter(lambda k: "cross" in k, attentions)), total=6, ncols=100):
        plot_attentions(i, img_data, results[0], label[0].cpu().numpy(), layer_id,
                        recorded_sampling_locations[layer][0][0].cpu().numpy(),
                        attentions[layer][0][0].cpu().numpy()
                       )

# %%
