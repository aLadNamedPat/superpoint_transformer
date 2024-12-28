import hydra
from src.utils import init_config
import torch

# 1. Init config
device = 'cuda:0'
task_name = 'semantic'  # 'panoptic'
expe_name = 'tracks.yaml'
ckpt_path = '~/superpoint_transformer/logs/checkpoint.ckpt'
split_name = 'test'

cfg = init_config(overrides=[
    f"datamodule={task_name}/{expe_name}",
    f"ckpt_path={ckpt_path}",
    "datamodule.load_full_res_idx=True"
])

# 2. Instantiate datamodule/model
datamodule = hydra.utils.instantiate(cfg.datamodule)
datamodule.prepare_data()
datamodule.setup()

if split_name == 'train':
    dataset = datamodule.train_dataset
elif split_name == 'val':
    dataset = datamodule.val_dataset
elif split_name == 'test':
    dataset = datamodule.test_dataset
else:
    raise ValueError(f"Unknown split '{split_name}'")

model = hydra.utils.instantiate(cfg.model)
model = model._load_from_checkpoint(ckpt_path)
model = model.eval().to(device)

# 3. Grab the entire tile (index 0, for example)
nag = dataset[0]
nag = dataset.on_device_transform(nag.to(device))

# 4. Inference
with torch.no_grad():
    output = model(nag)

# 5. Attach predictions
nag[0].semantic_pred = output.voxel_semantic_pred(
    super_index=nag[0].super_index
)
# If panoptic is relevant:
# vox_y, vox_index, vox_obj_pred = output.voxel_panoptic_pred(nag[0].super_index)
# nag[0].obj_pred = vox_obj_pred

# 6. Visualization
nag.show(
    class_names=dataset.class_names,
    class_colors=dataset.class_colors,
    stuff_classes=dataset.stuff_classes,
    num_classes=dataset.num_classes,
    max_points=600000, 
    centroids=True,
    h_edge=True,
    title="Entire File Visualization",
    path="Visualized_File.html"
)

print("Done! The file entire_tile_viz.html has been saved to your EC2 instance.")
