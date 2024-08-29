import os
import imageio
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils.data_augment import get_random_perturbation

 
def get_data_train(image_path, img_rows, img_cols, aug_len = 10):
    filenames = []
    total = 0
    print()
    for dirName, subdirList, fileList in os.walk(image_path):
        for filename in fileList:
            if ".png" in filename.lower():
                name = filename.split('.')[0]
                if os.path.exists(os.path.join(image_path, name + '_ROI.png')):
                    if os.path.exists(os.path.join(image_path, name + '_MASK.png')):
                        filenames.append(name)
                        total += 1
                        print('\tDetected',total,'sample', end="\r")
    print('Total', total, 'sample data for train')
    print('Augmentation:', aug_len, 'x')
    print()

    print('Preprocessing sample data for train')
    pbar = tqdm(total = total)
    imags_temp = np.ndarray((total, img_rows, img_cols, 3), dtype=np.uint8)
    masks_temp = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    for i in range(total):
            imag = imageio.imread(os.path.join(image_path, filenames[i] + '.png'))
            imag = np.uint8(imag)
            imag = resize(imag, (img_rows, img_cols, 3), order = 0)
            imag = np.uint8(imag*255.)
            imags_temp[i] = imag
            
            ROI = imageio.imread(os.path.join(image_path, filenames[i] + '_ROI.png'))
            if np.ndim(ROI) > 2: 
                ROI = ROI[:,:,-1]
            MASK0 = imageio.imread(os.path.join(image_path, filenames[i] + '_MASK.png'))
            if np.ndim(MASK0) > 2: 
                MASK0 = MASK0[:,:,-1]
            mask = np.float32(MASK0>100) + np.float32(ROI>100)
            mask = np.uint8(mask / 2. * 255.)
            mask = resize(mask, (img_rows, img_cols), order = 0)
            
            mask = np.uint8(mask * 255.)
            #mask = np.float32(mask>64) + np.float32(mask>160)
            #mask = np.uint8(np.float32(mask) / 2. * 255.)
                        
            masks_temp[i] = mask

            pbar.update(1)
    pbar.close()
    masks_temp = np.expand_dims(masks_temp, axis = 3)
    
    imags = np.ndarray((total*aug_len, img_rows, img_cols, 3), dtype=np.uint8)
    masks = np.ndarray((total*aug_len, img_rows, img_cols, 1), dtype=np.uint8)
    #imags = np.expand_dims(imags, axis = 4)
    #masks = np.expand_dims(masks, axis = 4)
    print('Augmenting data for train')
    pbar = tqdm(total = total*aug_len)
    k = 0
    for i in range(total):
        imags[k] = imags_temp[i]
        masks[k] = masks_temp[i]
        k += 1
        pbar.update(1)
        for j in range(aug_len - 1):
            imag_p, mask_p = get_random_perturbation(imags_temp[i], masks_temp[i])
            imags[k] = imag_p
            masks[k] = mask_p
            k += 1
            pbar.update(1)
    pbar.close()
    return imags, masks, filenames

def get_data_test(image_path, img_rows, img_cols):
    print()
    filenames = []
    total = 0
    for dirName, subdirList, fileList in os.walk(image_path):
        for filename in fileList:
            if ".jpg" in filename.lower():
                name = filename.split('.')[0]
                filenames.append(name)
                total += 1
    print('Total', total, 'sample images to analyse')

    print()
    print('Preprocessing sample data for test')
    pbar = tqdm(total = total)
    imags_temp = np.ndarray((total, img_rows, img_cols, 3), dtype=np.uint8)
    for i in range(total):
            imag = imageio.imread(os.path.join(image_path, filenames[i] + '.jpg'))
            imag = np.uint8(imag)
            imag = resize(imag, (img_rows, img_cols, 3), order = 0)
            imag = np.uint8(imag*255.)
            imags_temp[i] = imag

            pbar.update(1)
    pbar.close()
    imags = imags_temp
    
    return imags, filenames