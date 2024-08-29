# %%
import os
import shutil
import numpy as np
from tqdm import tqdm
from skimage.io import imsave

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from UNet_model import get_unet
from get_dataset import get_data_train

# %%
image_path = './28_files_from_Redness_photos_for_Valeriy/dataset/'
project_name = 'Unet_train'

project_dir = './' + project_name
if not os.path.exists(project_dir):
    os.mkdir(project_dir)

data_dir = project_dir + '/train_data'
if os.path.exists(data_dir):    shutil.rmtree(data_dir)
os.mkdir(data_dir)

weight_dir = project_dir + '/weights'
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir) 

log_dir = project_dir + '/logs'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


# %%
print()
print('-'*30, 'Loading and preprocessing train data...','-'*30)
img_rows = 512
img_cols = 512
img_chns = 3
aug_len = 40
imags_train, masks_train, filenames = get_data_train(image_path, img_rows, img_cols, aug_len)

imags_train = np.float32(imags_train)
masks_train = np.float32(masks_train)
imags_train /= 255.  # scale masks to [0, 1]
masks_train /= 255.  # scale masks to [0, 1]

print()
print('-'*30, 'Saving preprocessed data(images and masks) to files...','-'*30)
imags = np.uint8(imags_train*255.)
masks = np.uint8(np.squeeze(masks_train*255., axis=3))

print('Saving images amd masks to show the reality')
count_processed = 0
pbar = tqdm(total = imags.shape[0])
for i in range(0, imags.shape[0]):
    j = np.uint16(i/aug_len)
    imsave(os.path.join(data_dir, filenames[j] + '_train_' + str(count_processed) + '.png'), imags[i])
    imsave(os.path.join(data_dir, filenames[j] + '_train_' + str(count_processed) + '_MASK.png'), masks[i])
    count_processed += 1
    pbar.update(1)
pbar.close()

# %%
print('-'*30, 'Creating and compiling model...','-'*30)
model = get_unet(img_rows, img_cols, img_chns)
# model = tf.keras.models.load_model('./models/model_for_eye_redness_analysis.h5')
#model.summary()

if os.path.exists(os.path.join(weight_dir, project_name + '_aug.h5')):
    model.load_weights(os.path.join(weight_dir, project_name + '_aug.h5'))
    print('\n Weights loaded successfully \n')

model_checkpoint = ModelCheckpoint(
    os.path.join(weight_dir, project_name + '_aug.h5'), 
    monitor='val_loss', 
    save_best_only=True  
    )

csv_logger = CSVLogger(
    os.path.join(log_dir,  project_name + '_aug.txt'), 
    separator=',', 
    append=False)


print('-'*30,'Fitting model...','-'*30)
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
