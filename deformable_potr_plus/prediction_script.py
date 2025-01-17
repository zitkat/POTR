
"""
Standalone annotations script for hand pose estimation using DETR HPOES.

! Warning !
The script expects the input data in a .H5 dataset (under "images" key) in a form specified below. For other data
structures, please implement your own logic. Either way, the `depth_maps` should be an array of size (N, 256, 256).
"""

import argparse
import torch
import io
import h5py

import numpy as np

from deformable_potr_plus.models.deformable_potr_plus import DeformablePOTR
from deformable_potr_plus.models.backbone import build_backbone
from deformable_potr_plus.models.deformable_transformer import build_deformable_transformer

from dataset.dataset_processing import aug_morph_close

# Arguments
parser = argparse.ArgumentParser("DETR HPOES Standalone Annotations Script", add_help=False)
parser.add_argument("--weights_file", type=str, default="out/checkpoint.pth",
                    help="Path to the pretrained model's chekpoint (.pth)")
parser.add_argument("--input_file", type=str, default="in.h5",
                    help="Path to the .h5 file with input data (depth maps)")
parser.add_argument("--output_file", type=str, default="out.h5", help="Path to the .h5 file to write into")
parser.add_argument("--device", default="cpu", help="Device to be used")
parser.add_argument("--tta", default=1, help="Whether to use Test Time Augmentation")
args = parser.parse_args()

device = torch.device(args.device)

# Load the input data and checkpoint
print("Loading the input data and checkpoints.")
input_datafile = h5py.File(args.input_file, "r")
output_datafile = h5py.File(args.output_file, 'w')
checkpoint = torch.load(args.weights_file, map_location=device)

depth_maps_list = []
output_list = []

# Depth maps loading logic
num_img = len(input_datafile["images"])
# num_img = 1000
for record_index in range(num_img):
    print(record_index)
    pdata = input_datafile["images"][str(record_index)][:].tostring()
    _file = io.BytesIO(pdata)
    data = np.load(_file)["arr_0"]
    depth_maps_list.append(data)

depth_maps = np.asarray(depth_maps_list)
print("Structured input successfully.")

# Construct the model from the loaded data
model = DeformablePOTR(
    build_backbone(checkpoint["args"]),
    build_deformable_transformer(checkpoint["args"]),
    num_queries=checkpoint["args"].num_queries,
    num_feature_levels=checkpoint["args"].num_feature_levels
)
model.load_state_dict(checkpoint["model"])
model.to(device)

print("Constructed model successfully.")

# Iterate over the depth maps and structure the predictions
for i, depth_map in enumerate(depth_maps):
    # test time augmentation - morphological close
    if args.tta == 1:
        depth_map = aug_morph_close(depth_map)

    dm_tensor = torch.from_numpy(depth_map)
    dm_unsqueezed = dm_tensor.unsqueeze(0).expand(3, 224, 224).to(device, dtype=torch.float32)

    results = model([dm_unsqueezed]).detach().cpu().numpy()
    output_list.append(results)

print("Predictions were made.")

output = np.asarray(output_list).squeeze()
output_datafile.create_dataset("estimated_hands", data=output)
output_datafile.close()

print("Data was successfully structured and saved.")
