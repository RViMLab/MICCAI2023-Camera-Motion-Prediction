import os
import pandas as pd
from dotmap import DotMap

from utils.io import load_yaml
from utils.io import load_pickle, save_pickle
from utils.transforms import any_dict_list_to_compose
from lightning_data_modules import VideoDataModule

server = 'local'
server = DotMap(load_yaml('/home/martin/Dev/homography_imitation_learning/config/servers.yml')[server])

meta_df_name = 'cholec80_dummy_transforms.pkl'
# meta_df_name = 'cholec80_retransfor ms.pkl'
meta_df = pd.read_pickle(os.path.join(server.config.location, meta_df_name))
meta_df.aug_transforms = None  # no transforms for evaluation

# load cholec80
prefix = server.database.location
clip_length_in_frames = 2
frames_between_clips = 1
frame_rate = 5
train_split = 0.5
batch_size = 1
num_workers = 8
random_state = 42

# None if not initialized
# test_md_name = 'cholec80_homography_regression_discrepancy_test_md_frame_rate_{}.pkl'.format(frame_rate)
test_md_name = 'cholec80_dummy_test_md_frame_rate_{}.pkl'.format(frame_rate)

try:
    test_md = load_pickle(os.path.join(server.config.location, test_md_name))
except:
    test_md = None

dm = VideoDataModule(
    meta_df,
    prefix=prefix,
    clip_length_in_frames=clip_length_in_frames,
    frames_between_clips=frames_between_clips   ,
    frame_rate=frame_rate,
    train_split=train_split,
    batch_size=batch_size,
    num_workers=num_workers,
    random_state=random_state,
    test_metadata=test_md
)

_, _, test_md = dm.setup('test')

# store metadata
if test_md:
    save_pickle(os.path.join(server.config.location, test_md_name), test_md)



import cv2
import numpy as np
from tqdm import tqdm
import torch
from kornia.geometry import warp_perspective, tensor_to_image
import matplotlib.pyplot as plt

from lightning_modules import DeepImageHomographyEstimationModuleBackbone
from utils.processing import four_point_homography_to_matrix, image_edges
from utils.processing import FeatureHomographyEstimation
from utils.viz import yt_alpha_blend
from utils.io import generate_path

# feature-based model creationg stage
fd = cv2.SIFT_create()
fh = FeatureHomographyEstimation(fd)

# deep creationg stage
prefix = '/home/martin/Tresors/homography_imitation_learning_logs/deep_image_homography_estimation_backbone_search/version_8'
configs = load_yaml(os.path.join(prefix, 'config.yml'))
model = DeepImageHomographyEstimationModuleBackbone.load_from_checkpoint(os.path.join(prefix, 'checkpoints/epoch=99-step=44099.ckpt'), shape=configs['model']['shape'])

device = 'cpu'
if torch.cuda.is_available():
    print('Running with CUDA backend.')
    device = 'cuda'

model.to(device)
model = model.eval()
model.freeze()

# data iterator
test_dl = dm.test_dataloader()

batch = next(iter(test_dl))
batch = dm.transfer_batch_to_device(batch, device)
vid, aug_vid, frame_rate, video_fps, video_idx, idx = batch

ones = torch.ones(vid.shape, device=vid.device, requires_grad=True)

img, wrp = vid[0,0].unsqueeze(0), vid[0,1].unsqueeze(0)
duv_target = model(img, wrp).detach()


epochs = 100

opt = torch.optim.Adam([ones], lr=10.)
loss = torch.nn.MSELoss()

for epoch in range(epochs):
    opt.zero_grad()
    vid_masked = vid*ones

    img, wrp = vid_masked[0,0].unsqueeze(0), vid_masked[0,1].unsqueeze(0)
    duv = model(img, wrp)

    l_duv = loss(duv, duv_target)
    reg = ones.sum()
    lam = duv.numel()/ones.numel()
    print(epoch, l_duv)
    l = 100*l_duv + lam*reg
    l.backward()
    opt.step()


vid_masked = vid*ones
img, wrp = tensor_to_image(vid_masked[0,0]), tensor_to_image(vid_masked[0,1])
cv2.imshow('img', img[...,::-1])
cv2.imshow('wrp', wrp[...,::-1])
cv2.waitKey()







