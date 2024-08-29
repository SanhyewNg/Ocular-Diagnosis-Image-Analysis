import os
import imageio
import numpy as np
from tqdm import tqdm
from skimage.io import imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

# %%

img_rows = 512
img_cols = 512
img_chns = 3

image_path = 'datasets/LG_Staining'
pred_dir = 'datasets_preprocessed/LG_Staining'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

# %% 
filenames = []
total = 0
print()
for dirName, subdirList, fileList in os.walk(image_path):
    for filename in fileList:
        if ".jpg" in filename.lower():
            name = filename.split('.')[0]
            if os.path.exists(os.path.join(image_path, name + '_ROI.jpg')):
                if os.path.exists(os.path.join(image_path, name + '_MASK.jpg')):
                    filenames.append(name)
                    total += 1
                    print('\tDetected',total,'sample', end="\r")
print('Total', total, 'jpg sample data for train')
print()

print('Preprocessing sample data for train')
pbar = tqdm(total = total)
imags_temp = np.ndarray((total, img_rows, img_cols, 3), dtype=np.uint8)
masks_temp = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
rois_temp = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)

for i in range(total):
    imag = imageio.imread(os.path.join(image_path, filenames[i] + '.jpg'))
    imag = np.uint8(imag)
    imag = resize(imag, (img_rows, img_cols, 3), order = 0)
    imag = np.uint8(imag*255.)
    imags_temp[i] = imag
    
    mask = imageio.imread(os.path.join(image_path, filenames[i] + '_MASK.jpg'))
    if np.ndim(mask) > 2: 
        mask = mask[:,:,-1]
    mask = np.float32(mask>100)
    mask = np.uint8(mask * 255.)
    mask = resize(mask, (img_rows, img_cols), order = 0)
    mask = np.uint8(mask * 255.)
    masks_temp[i] = mask

    roi = imageio.imread(os.path.join(image_path, filenames[i] + '_ROI.jpg'))
    if np.ndim(roi) > 2: 
        roi = roi[:,:,-1]
    roi = np.float32(roi>100)
    roi = np.uint8(roi * 255.)
    roi = resize(roi, (img_rows, img_cols), order = 0)
    roi = np.uint8(roi * 255.)
    rois_temp[i] = roi

    imsave(os.path.join(pred_dir, filenames[i] + '.png'), imag)
    imsave(os.path.join(pred_dir, filenames[i] + '_MASK.png'), mask)
    imsave(os.path.join(pred_dir, filenames[i] + '_ROI.png'), roi)

    pbar.update(1)
pbar.close()
