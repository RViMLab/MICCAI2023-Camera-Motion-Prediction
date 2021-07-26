# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Create Dataloader

# %%
import os
import pandas as pd
from dotmap import DotMap

from utils.io import load_yaml
from utils.io import load_pickle, save_pickle
from utils.transforms import anyDictListToCompose
from lightning_data_modules import VideoDataModule

server = 'local'
server = DotMap(load_yaml('../config/servers.yml')[server])

meta_df_name = 'cholec80_transforms.pkl'
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
test_md_name = 'cholec80_homography_regression_discrepancy_test_md_frame_rate_{}.pkl'.format(frame_rate)

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

# %% [markdown]
# # Evaluate Feature-based Homography Estimation

# %%
import cv2
import numpy as np
import torch
from kornia import warp_perspective, tensor_to_image

from lightning_modules import DeepImageHomographyEstimationModuleBackbone
from utils.processing import four_point_homography_to_matrix, image_edges
from utils.processing import FeatureHomographyEstimation
from utils.viz import yt_alpha_blend
from utils.io import generate_path

# feature-based model creationg stage
fd = cv2.xfeatures2d.SIFT_create()
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

# log dataframe
log_df = pd.DataFrame(columns=['frame_rate', 'video_fps', 'video_idx', 'idx', 'duv'])

# data iterator
test_dl = dm.test_dataloader()

hist = []

for batch in test_dl:
    # deep
    batch = dm.transfer_batch_to_device(batch, device)
    vid, aug_vid, frame_rate, video_fps, video_idx, idx = batch

    img_deep, wrp_deep = vid[0,0].unsqueeze(0), vid[0,1].unsqueeze(0)

    duv_deep = model(img_deep, wrp_deep)
    duv_deep *= -1

    uv_deep = image_edges(img_deep)
    H_deep = four_point_homography_to_matrix(uv_deep, duv_deep)

    # classical
    batch = dm.transfer_batch_to_device(batch, 'cpu')
    vid, aug_vid, frame_rate, video_fps, video_idx, idx = batch

    img, wrp = tensor_to_image(vid[0,0]), tensor_to_image(vid[0,1])

    H, duv = fh((img*255).astype(np.uint8), (wrp*255).astype(np.uint8))
    
    log_df = log_df.append({
        'frame_rate': frame_rate.numpy(), 'video_fps': np.array([x.numpy() for x in video_fps]), 'video_idx': video_idx.numpy(), 'idx': idx.numpy(), 'duv': duv
    }, ignore_index=True)

    if H is not None:
        hist.append(np.linalg.norm(duv - duv_deep.cpu().numpy().squeeze(), axis=1).mean())
        # # deep
        # wrp_deep_reg = warp_perspective(img_deep, H_deep, (img_deep.shape[-2], img_deep.shape[-1]))

        # blend_deep = yt_alpha_blend(wrp_deep, wrp_deep_reg)

        # blend_deep = tensor_to_image(blend_deep)
        # cv2.imshow('blend_deep', blend_deep)
    
        # # classical
        # wrp_reg = cv2.warpPerspective(img, H, (wrp.shape[1], wrp.shape[0]))

        # blend = yt_alpha_blend(wrp, wrp_reg)

        # cv2.imshow('blend', blend)
        # cv2.waitKey()

# log_path = os.path.join(server.logging.location, 'homography_regression_discrepancy')
# generate_path(log_path)
# log_df.to_pickle(os.path.join(log_path, 'feature_based_frame_rate_{}.pkl'.format(frame_rate.item())))


# %%
import matplotlib.pyplot as plt

plt.hist(hist, bins=50)
plt.show()

# %% [markdown]
# # Evaluate Deep Homography Estimation

# %%
import cv2
import torch
from kornia import warp_perspective, tensor_to_image

from lightning_modules import DeepImageHomographyEstimationModuleBackbone
from utils.processing import four_point_homography_to_matrix, image_edges
from utils.viz import yt_alpha_blend
from utils.io import generate_path

# model creationg stage
prefix = '/home/martin/Tresors/homography_imitation_learning_logs/deep_image_homography_estimation_backbone_search/version_6'
configs = load_yaml(os.path.join(prefix, 'config.yml'))
model = DeepImageHomographyEstimationModuleBackbone.load_from_checkpoint(os.path.join(prefix, 'checkpoints/epoch=52-step=46692.ckpt'), shape=configs['model']['shape'])

device = 'cpu'
if torch.cuda.is_available():
    print('Running with CUDA backend.')
    device = 'cuda'

model.to(device)
model = model.eval()
model.freeze()

# log dataframe
log_df = pd.DataFrame(columns=['frame_rate', 'video_fps', 'video_idx', 'idx', 'duv'])

# data iterator
test_dl = dm.test_dataloader()

for batch in test_dl:
    batch = dm.transfer_batch_to_device(batch, device)
    vid, aug_vid, frame_rate, video_fps, video_idx, idx = batch

    img, wrp = vid[0,0].unsqueeze(0), vid[0,1].unsqueeze(0)

    duv = model(img, wrp)
    duv *= -1

    log_df = log_df.append({
        'frame_rate': frame_rate.cpu().numpy(), 'video_fps': np.array([x.cpu().numpy() for x in video_fps]), 'video_idx': video_idx.cpu().numpy(), 'idx': idx.cpu().numpy(), 'duv': duv.cpu().numpy()
    }, ignore_index=True)

    # uv = image_edges(img)
    # H = four_point_homography_to_matrix(uv, duv)

    # wrp_reg = warp_perspective(img, H, (img.shape[-2], img.shape[-1]))

    # blend = yt_alpha_blend(wrp, wrp_reg)

    # blend = tensor_to_image(blend)
    # cv2.imshow('blend', blend)
    # cv2.waitKey()

log_path = os.path.join(server.logging.location, 'homography_regression_discrepancy')
generate_path(log_path)
log_df.to_pickle(os.path.join(log_path, 'deep_frame_rate_{}.pkl'.format(frame_rate.item())))

# iterate through test set

# estimate homography, store duv, idx -> mangage to load frame by index


# %%
feature_df = pd.read_pickle('/media/martin/Samsung_T5/logs/homography_regression_discrepancy/feature_based_frame_rate_5.pkl')
deep_df = pd.read_pickle('/media/martin/Samsung_T5/logs/homography_regression_discrepancy/deep_frame_rate_5.pkl')

# diff = feature_df.duv - deep_df.duv

# print(diff)


# %%



