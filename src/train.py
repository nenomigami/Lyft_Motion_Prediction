#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:38:46 2020

@author: hoyun
"""
from src.common_import import *
from model.Resnet18 import *
from src.utils import FixedShuffleDataset
# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'data_path': "../data",
    'model_params': {
        'model_architecture': 'resnet18',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "model_resnet18",
        'lr': 1e-3,
        'weight_path': None,
         #'weight_path': "../results/model_resnet18_25k_output_670000.pth",
         #'checkpoint_path': None
         'checkpoint_path': "model_resnet18_100.pth"
    },

    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5,
        'disable_traffic_light_faces' : False,
    },

    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 32, #resnet50
        'shuffle': False,
        'num_workers': 8
    },
    
    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 8
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 8
    },

    'train_params': {
        'max_num_steps': 500,
        'checkpoint_every_n_steps': 100,
    }
}
#set random seed
seed = 42
torch.manual_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
# set env variable for data
DIR_INPUT = cfg["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)

weight_path = cfg["model_params"]["weight_path"]
checkpoint_path = cfg["model_params"]["checkpoint_path"]

# ===== INIT TRAIN DATASET============================================================
train_cfg = cfg["train_data_loader"]
if checkpoint_path:
    step = torch.load(checkpoint_path)["step"]
else: 
    step = 0
rasterizer = build_rasterizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open(cached=False)
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataset = FixedShuffleDataset(train_dataset, step*train_cfg["batch_size"])
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])

print("==================================TRAIN DATA==================================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LyftMultiModel(cfg)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])

#load weight if there is a pretrained model
if weight_path:
    model.load_state_dict(torch.load(weight_path))
if checkpoint_path:
    model.load_state_dict(torch.load(checkpoint_path)["weight"])
    optimizer.load_state_dict(torch.load(checkpoint_path)["optimizer"])

print(f'device {device}')

# ==== TRAINING LOOP =========================================================


tr_it = iter(train_dataloader)
num_iter = cfg["train_params"]["max_num_steps"] - step
progress_bar = tqdm(range(num_iter))
losses_train = []
losses_per_100 = []
iterations = []
metrics = []
times = []
model_name = cfg["model_params"]["model_name"]
start = time.time()
for i in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    model.train()
    torch.set_grad_enabled(True)

    loss, _ = forward(data, model, device)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
 
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
    if (i+1) % (cfg['train_params']['checkpoint_every_n_steps']) == 0:
        torch.save({"step" : i+1,
                    "weight" : model.state_dict(),
                    "optimizer" : optimizer.state_dict()}, f'{model_name}_{i+1}.pth')
        iterations.append(i+1)
        metrics.append(np.mean(losses_train))
        losses_per_100.append(np.array(losses_train)[-100:].mean())
        times.append((time.time()-start)/60)
        
        results = pd.DataFrame({'iterations': iterations, 'metrics (avg)': metrics, 
                                'losses_per_100' : losses_per_100, 'elapsed_time (mins)': times})
        results.to_csv(f"train_metrics_{model_name}_{i+1}.csv", index = False)
print(f"Total training time is {(time.time()-start)/60} mins")
print(results)
