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

project_name = 'Lower_Eyelid'
#project_name = 'Upper_Eyelid'
image_path = './Eyelid Image Staining Analysis/' + project_name + '_dataset/'

project_dir = './Eyelid Image Staining Analysis/' + project_name
if not os.path.exists(project_dir):
    os.mkdir(project_dir)

pred_dir = project_dir + '/dataset_preprocessed'
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
            if os.path.exists(os.path.join(image_path, name + '_A_MASK.jpg')):
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
    
    mask = imageio.imread(os.path.join(image_path, filenames[i] + '_A_MASK.jpg'))
    if np.ndim(mask) > 2: 
        mask = mask[:,:,-1]
    mask = np.float32(mask>100)
    mask = np.uint8(mask * 255.)
    mask = resize(mask, (img_rows, img_cols), order = 0)
    mask = np.uint8(mask * 255.)
    masks_temp[i] = mask

    imsave(os.path.join(pred_dir, filenames[i] + '_512.png'), imag)
    imsave(os.path.join(pred_dir, filenames[i] + '_512_MASK.png'), mask)

    pbar.update(1)
pbar.close()



