import time
import os
import torch
import monai
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from options.test_options import TestOptions
from models.models import create_model
from pdb import set_trace as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# from data.dataset_TEST import CreateDataloader_TEST
from data.dataset_CACHE_Whole_body import CreateDataloader  # data.dataset_CACHE
# from data.dataset_CACHE import CreateDataloader
from monai.transforms import SpatialCropd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import imageio
from scipy.ndimage import gaussian_filter
import pandas as pd
from tqdm import tqdm
import os, json

import time


patch_dim = 32
overlap_dim = 16
# Calculates 3D patch centers to subdivide an image into smaller portions with overlapping
def patch_indices(image_shape, patch_size=(patch_dim, patch_dim, patch_dim), overlap=(overlap_dim, overlap_dim, overlap_dim)):
    indices = []
    step_size = np.subtract(patch_size, overlap)

    for x in range(0, image_shape[0], step_size[0]):
        for y in range(0, image_shape[1], step_size[1]):
            for z in range(0, image_shape[2], step_size[2]):
                center = (x + patch_size[0] // 2, y + patch_size[1] // 2, z + patch_size[2] // 2)

                # Make sure the center does not exceed the size of the image.
                center = (
                    min(center[0], image_shape[0] - overlap[0]),
                    min(center[1], image_shape[1] - overlap[1]),
                    min(center[2], image_shape[2] - overlap[2])
                )

                indices.append(center)

    return indices

opt = TestOptions().parse()  # set phase == test


#Help functions to write the inference progress to the stdout, which is captured by the parent process
def write_progress(data):
    print("\n::PROGRESS:: " + json.dumps(data) + "\n", flush=True)

def update_progress(job_id, step, total):
    data = {
        "step": step,
        "total": total,
        "job_id": job_id,
        "finished": False
    }
    write_progress(data)

def write_finished(job_id):
    data = {
        "step": 1,
        "total": 1,
        "job_id": job_id,
        "finished": True
    }
    write_progress(data)
        

opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

folder_paths= []
folder_paths.append(opt.upload_dir)

data_loader = CreateDataloader(opt, shuffle=False, cache=False ,folder_paths=folder_paths )  # CreateDataloader_TEST
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

model = create_model(opt)
#visualizer = Visualizer(opt)

mae_per_patient = []
psnr_per_patient = []
ssim_per_patient = []


# test
for i, data in enumerate(data_loader):

    image_shape = data['A'].shape[2:]
    patch_centers = patch_indices(image_shape)
    #print(len(patch_centers))
    # Initialize a numpy array for the full generated image.
    complete_generated_image = np.zeros_like(data['B'].as_tensor().permute(0, 2, 3, 4, 1))

    # Initialize counter to keep track of the number of overlapping patches at each point.
    overlap_counter = np.zeros_like(data['B'].as_tensor().permute(0, 2, 3, 4, 1))

    for i, center in enumerate(tqdm(patch_centers)):

        spatial_crop = SpatialCropd(keys=["A", "B"], roi_center=center, roi_size=[patch_dim, patch_dim, patch_dim])
        patches = spatial_crop({"A": data['A'].as_tensor().permute(0, 2, 3, 4, 1), "B": data['B'].as_tensor().permute(0, 2, 3, 4, 1)})

        patches['A'] = patches['A'].squeeze().unsqueeze(0).unsqueeze(0)
        patches['B'] = patches['B'].squeeze().unsqueeze(0).unsqueeze(0)

        model.set_input(patches)
        model.test()
        visuals = model.get_current_visuals()  # returns a dictionary of images: real_A, fake_B, real_B


        real_B_test = visuals['real_B']
        fake_B_test = visuals['fake_B']

        complete_generated_image[0,
        center[0] - overlap_dim:center[0] + overlap_dim,
        center[1] - overlap_dim:center[1] + overlap_dim,
        center[2] - overlap_dim:center[2] + overlap_dim,
        0] += fake_B_test.squeeze()

        overlap_counter[0,
        center[0] - overlap_dim:center[0] + overlap_dim,
        center[1] - overlap_dim:center[1] + overlap_dim,
        center[2] - overlap_dim:center[2] + overlap_dim,
        0] += 1

        # Convert to tensors
        fake_B_test = torch.tensor(fake_B_test)
        real_B_test = torch.tensor(real_B_test)

        if (i % 10 == 0) or (i >= len(patch_centers) - 3):
            update_progress(opt.json_id, i, len(patch_centers))
       

    # IMAGE RECONSTRUCTION: Calculates the final average by dividing by the number of overlapping patches.
    complete_generated_image /= np.maximum(overlap_counter, 1)

    # Save the full generated image in the out_path folder.
    out_path = opt.out_path
    os.makedirs(out_path, exist_ok=True)

    image_filename_nifti = f"{opt.json_id}.nii.gz"
    image_path_nifti = os.path.join(out_path, image_filename_nifti)

    complete_generated_image_3d = complete_generated_image.squeeze(axis=(0, 4))

    complete_generated_image_nifti = nib.Nifti1Image(complete_generated_image_3d, np.eye(4))
    nib.save(complete_generated_image_nifti, image_path_nifti)

    write_finished(opt.json_id)