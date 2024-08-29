# %%
import os
import imageio
from tqdm import tqdm

import numpy as np
from skimage.io import imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# %%
model_path = './LG Staining Capture Analysis/models/model_for_LG_staining_analysis.h5'

image_path = './LG Staining Capture Analysis/Model_training_part2/'
pred_dir = './LG Staining Capture Analysis/dataset_preprocessed'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

img_rows = 512
img_cols = 512
img_chns = 3

def get_data_correction(image_path, img_rows, img_cols):
    filenames = []
    total = 0
    print()
    for dirName, subdirList, fileList in os.walk(image_path):
        for filename in fileList:
            if ".png" in filename.lower():
                name = filename.split('.')[0]
                if os.path.exists(os.path.join(image_path, name + '_analysed_A2.png')):
                    filenames.append(name)
                    total += 1
                    print('\tDetected', total, 'sample', end="\r")
    print('Total', total, 'image pairs for correction')
    print()

    print('Reading and resizing image pairs for correction')
    pbar = tqdm(total = total)
    imags = np.ndarray((total, img_rows, img_cols, 3), dtype=np.uint8)
    imags_analysed_A2 = np.ndarray((total, img_rows, img_cols, 3), dtype=np.uint8)
    for i in range(total):
            imag = imageio.imread(os.path.join(image_path, filenames[i] + '.png'))
            imag = np.uint8(imag)
            imag = resize(imag, (img_rows, img_cols, 3), order = 0)
            imag = np.uint8(imag*255.)
            imags[i] = imag
            
            imag = imageio.imread(os.path.join(image_path, filenames[i] + '_analysed_A2.png'))
            imag = np.uint8(imag)
            imag = resize(imag, (img_rows, img_cols, 3), order = 0)
            imag = np.uint8(imag*255.)
            imags_analysed_A2[i] = imag
            
            pbar.update(1)
    pbar.close()
    
    return imags, imags_analysed_A2, filenames

# %%
print()
print('-'*30, 'Detecting and Loading image pairs to correct...','-'*30)
imags, imags_analysed_A2, filenames = get_data_correction(image_path, img_rows, img_cols)

# %%
imags_test = np.float32(imags)
imags_test /= 255.

print()
print('-'*30, 'Loading model...','-'*30)
model = load_model(model_path)

print()
print('-'*30, 'Predicting masks on test data...','-'*30)
masks_test = model.predict(imags_test, batch_size=1, verbose=1)


# %%
N_images = imags_test.shape[0]

masks_test = np.float32(np.squeeze(masks_test, axis=3))

ROIs = np.float32(masks_test>0.25)
ROIs = np.uint8(ROIs * 255.)

MASKs = np.float32(masks_test>0.75)
MASKs = np.uint8(MASKs * 255.)

# %%
print()

def ColorRangeMap(imag, RL, RH, GL, GH, BL, BH):
    R = np.minimum(imag[:,:,0] >= RL, imag[:,:,0] < RH)
    G = np.minimum(imag[:,:,1] >= GL, imag[:,:,1] < GH)
    B = np.minimum(imag[:,:,2] >= BL, imag[:,:,2] < BH)
    Z = np.minimum(np.minimum(R,G), B)
    return Z

crm1 = ColorRangeMap(imags_analysed_A2[46],   0,  50,  0, 50, 200, 256)
crm2 = ColorRangeMap(imags_analysed_A2[46], 250, 256, 60, 68, 186, 196)
crm = np.maximum(crm1, crm2)


plt.imshow(np.uint8(crm)*255, cmap ='Greens')
plt.show()

print()

print('Saving masks')
pbar = tqdm(total = imags_analysed_A2.shape[0])
for i in range(0, imags_analysed_A2.shape[0]):

    crm1 = ColorRangeMap(imags_analysed_A2[i],   0,  50,  0, 50, 200, 256)
    crm2 = ColorRangeMap(imags_analysed_A2[i], 250, 256, 60, 68, 186, 196)
    crm = np.maximum(crm1, crm2)

    imsave(os.path.join(pred_dir, filenames[i] + '_MASK.png'), crm)

    imsave(os.path.join(pred_dir, filenames[i] + '.png'), imags[i])
    imsave(os.path.join(pred_dir, filenames[i] + '_ROI.png'), np.maximum(ROIs[i], crm))

    pbar.update(1)
pbar.close()