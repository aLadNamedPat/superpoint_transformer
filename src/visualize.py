import os
import sys

# Add the project's files to the python path
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
# file_path = os.path.dirname(os.path.abspath(''))  # for .ipynb notebook
sys.path.append(file_path)

import hydra
from src.utils import init_config
import torch
from src.transforms import *
from src.utils.widgets import *
from src.data import *

device_widget = make_device_widget()
task_widget, expe_widget = make_experiment_widgets()
split_widget = make_split_widget()
ckpt_widget = make_checkpoint_file_search_widget()

device_widget_value = "cuda"
task_widget_semantic = "semantic"
split_widget_value = "test"
expe_widget_value = "tracks"
ckpt_widget_value = "logs/epoch_120-v2.ckpt"
print(f"You chose:")
print(f"  - device={device_widget_value}")
print(f"  - task={task_widget_semantic}")
print(f"  - split={split_widget_value}")
print(f"  - experiment={expe_widget_value}")
print(f"  - ckpt={ckpt_widget_value}")

cfg = init_config(overrides=[
    f"datamodule={task_widget_semantic}/{expe_widget_value}",
    f"ckpt_path={ckpt_widget_value}",
    f"datamodule.load_full_res_idx={True}"  # only when you need full-resolution predictions 
])

datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.prepare_data()
datamodule.setup()

if split_widget_value == 'train':
    dataset = datamodule.train_dataset
elif split_widget_value == 'val':
    dataset = datamodule.val_dataset
elif split_widget_value == 'test':
    dataset = datamodule.test_dataset
else:
    raise ValueError(f"Unknown split '{split_widget_value}'")

dataset.print_classes()

# Instantiate the model
model = hydra.utils.instantiate(cfg.model)

# Load pretrained weights from a checkpoint file
if ckpt_widget.value is not None:
    model = model._load_from_checkpoint(cfg.ckpt_path)

# Move model to selected device
model = model.eval().to(device_widget_value)

for t in dataset.on_device_transform.transforms:
    if isinstance(t, NAGAddKeysTo):
        t.delete_after = False

# Load the first dataset item. This will return the hierarchical 
# partition of an entire tile, as a NAG object 
nag = dataset[0]

# Apply on-device transforms on the NAG object. For the train dataset, 
# this will select a spherical sample of the larger tile and apply some
# data augmentations. For the validation and test datasets, this will
# prepare an entire tile for inference
nag = dataset.on_device_transform(nag.to(device_widget_value))

# Inference, returns a task-specific ouput object carrying predictions
with torch.no_grad():
    output = model(nag)

# Compute the level-0 (voxel-wise) semantic segmentation predictions 
# based on the predictions on level-1 superpoints and save those for 
# visualization in the level-0 Data under the 'semantic_pred' attribute
nag[0].semantic_pred = output.voxel_semantic_pred(super_index=nag[0].super_index)

# Predefined radius and center locations for each dataset
# Feel free to modify these values
center = nag[0].pos.mean(dim=0).view(1, -1)
if 'dales' in expe_widget.value:
    radius = 10
elif 'kitti360' in expe_widget.value:
    radius = 10
elif 'scannet' in expe_widget.value:
    radius = 10
elif 's3dis' in expe_widget.value:
    radius = 3
else:
    radius = 3

nag.show(
    figsize=1600,
    radius=radius,
    center=center,
    class_names=dataset.class_names,
    class_colors=dataset.class_colors,
    stuff_classes=dataset.stuff_classes,
    num_classes=dataset.num_classes,
    max_points=100000,
    title="My Interactive Visualization Partition", 
    path="my_interactive_visualization.html"
)