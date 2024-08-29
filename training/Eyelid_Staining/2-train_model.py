# %%
import os
from tqdm import tqdm

import numpy as np
from skimage.io import imsave

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from utils.UNet_model import get_unet
from utils.get_dataset import get_data_train

# %%
#project_name = 'Eyeball'
project_name = 'Eyelid'
#project_name = 'Lower_Eyelid'
#project_name = 'Upper_Eyelid'
project_dir = './Eyelid Image Staining Analysis/' + project_name
image_path = project_dir + '/dataset_preprocessed/'
data_dir = project_dir + '/train_data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
weight_dir = project_dir + '/weights'
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)
#weight_filename = project_name + '_from_Lower_and_Upper'
weight_filename = project_name + '_from_scratch'
log_dir = project_dir + '/logs'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


img_rows = 512
img_cols = 512
img_chns = 3

aug_len = 10
# %%
print()
print('-'*30, 'Loading and preprocessing train data','-'*30)
imags_train, masks_train, filenames = get_data_train(image_path, img_rows, img_cols, aug_len)

imags_train = np.float32(imags_train)
masks_train = np.float32(masks_train)
imags_train /= 255.  # scale masks to [0, 1]
masks_train /= 255.  # scale masks to [0, 1]

# print()
# print('-'*30, 'Saving preprocessed data(images and masks) to files','-'*30)
# imags = np.uint8(imags_train*255.)
# masks = np.uint8(np.squeeze(masks_train*255., axis=3))

# print('Saving images')
# count_processed = 0
# pbar = tqdm(total = imags.shape[0])
# for i in range(0, imags.shape[0]):
#     j = np.uint16(i/aug_len)
#     imsave(os.path.join(data_dir, filenames[j] + '_train_' + str(count_processed) + '.png'), imags[i])
#     count_processed += 1
#     pbar.update(1)
# pbar.close()

# print('Saving masks')
# count_processed = 0
# pbar = tqdm(total = masks.shape[0])
# for i in range(0, masks.shape[0]):
#     j = np.uint16(i/aug_len)
#     imsave(os.path.join(data_dir, filenames[j] + '_train_' + str(count_processed) + '_MASK.png'), masks[i])
#     count_processed += 1
#     pbar.update(1)
# pbar.close()

# %%
print('-'*30, 'Creating and compiling model','-'*30)
model = get_unet(img_rows, img_cols, img_chns)
#model.summary()

if os.path.exists(os.path.join(weight_dir, weight_filename + '.h5')):
    model.load_weights(os.path.join(weight_dir, weight_filename + '.h5'))
    print('\n Weights loaded successfully \n')

model_checkpoint = ModelCheckpoint(
    os.path.join(weight_dir, weight_filename + '.h5'), 
    monitor='val_loss', 
    save_best_only=True  
    )

csv_logger = CSVLogger(
    os.path.join(log_dir,  weight_filename + '.txt'), 
    separator=',', 
    append=False)


print('-'*30,'Fitting model','-'*30)
model.fit(
    imags_train,
    masks_train,
    batch_size=5,
    epochs=5000,
    verbose=1,
    shuffle=True,
    validation_split=0.10,
    callbacks=[model_checkpoint, csv_logger]
    )

print('-'*30,'Training finished','-'*30)
