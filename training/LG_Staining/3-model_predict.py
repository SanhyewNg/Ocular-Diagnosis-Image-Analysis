# %%
import os
import numpy as np
from tqdm import tqdm
from skimage.io import imsave

from utils.UNet_model import get_unet
from utils.get_dataset import get_data_test


# %%
project_dir = 'training/LG_Staining'
image_dir =  project_dir + '/test_images'   
result_dir = project_dir + '/test_results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

img_rows = 512
img_cols = 512
img_chns = 3


# %%
print()
print('-'*30, 'Loading and preprocessing test data...','-'*30)

imags_test, filenames = get_data_test(image_dir, img_rows, img_cols)
imags_test = np.float32(imags_test)
imags_test /= 255.  # scale masks to [0, 1]


# %%
print()
print('-'*30, 'Saving preprocessed test images to files...','-'*30)

imags = np.uint8(imags_test*255.)

count_processed = 0
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
pbar = tqdm(total = imags.shape[0])
for i in range(0, imags.shape[0]):
    #imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imags[i])
    imsave(os.path.join(result_dir,  filenames[i] + '.png'), imags[i])
    count_processed += 1
    
    pbar.update(1)
pbar.close()



# %%
print()
print('-'*30, 'Creating and compiling model...','-'*30)
model = get_unet(img_rows, img_cols, img_chns)
#model.summary()



# %%
print()
print('-'*30, 'Loading saved weights...','-'*30)
weight_dir = project_dir + '/weights'
model.load_weights(os.path.join(weight_dir,  + '_aug.h5'))



# %%
model_dir = project_dir + '/models'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model.save(os.path.join(model_dir, 'model_for_LG_staining_analysis.h5'))



# %%
print()
print('-'*30, 'Predicting masks on test data...','-'*30)
masks_test = model.predict(imags_test, batch_size=1, verbose=1)



# %%
print()
print('-'*30, 'Saving predicted test masks to files...','-'*30)
imgs_mask_test = np.float32(np.squeeze(masks_test, axis=3))

imgs_mask_test = (np.float32(imgs_mask_test>0.25) + np.float32(imgs_mask_test>0.75)) / 2.
#imgs_mask_test = np.float32(imgs_mask_test>0.65)

imgs_mask_test = np.uint8(imgs_mask_test * 255.)
#imgs_mask_test = np.around(imgs_mask_test, decimals=0)

count_processed = 0
pbar = tqdm(total = imgs_mask_test.shape[0])
for i in range(0, imgs_mask_test.shape[0]):
    imsave(os.path.join(result_dir, filenames[i] + '_mask_predicted' + '.png'), imgs_mask_test[i])
    count_processed += 1

    pbar.update(1)
pbar.close()


# %%
print('-'*30, 'Saving analysed test results to files...','-'*30)

count_processed = 0
pbar = tqdm(total = imags.shape[0])
for i in range(0, imags.shape[0]):
    data = np.array(imags[i])
    white_areas = (imgs_mask_test[i] > 200)
    data[white_areas, :] = (255,63,192)

    imsave(os.path.join(result_dir,  filenames[i] + '_analysed' + '.png'), data)
    count_processed += 1

    pbar.update(1)
pbar.close()


# %%
