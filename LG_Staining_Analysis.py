# %%
import os
import imageio
import datetime
import numpy as np

from tqdm import tqdm
from openpyxl import Workbook

from skimage.io import imsave
from skimage.transform import resize

from tensorflow.keras.models import load_model

# %%
image_path = 'test_images/LG_Staining/'
model_dir = 'models'

current_time = datetime.datetime.now() 
time_str = current_time.strftime("%Y%m%d_%H%M%S")
pred_dir = 'results/LG_Staining_Analysis_Result_' + time_str
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
out_filename = pred_dir + '/LG_Staining_Analysis_Result_Values_' + time_str + '.xlsx'

img_rows = 512
img_cols = 512
img_chns = 3

area_image = 162.40566892215887900261068783633 # mm2

# %%
def get_data_test(image_path, img_rows, img_cols):
    print()
    filenames = []
    N_images = 0
    for dirName, subdirList, fileList in os.walk(image_path):
        for filename in fileList:
            if ".jpg" or ".png" in filename.lower():
                name = filename.split('.')[0]
                filenames.append(name)
                N_images += 1
    print('Total', N_images, 'sample images to analyse')

    print()
    print('Preprocessing sample data for test')
    pbar = tqdm(total = N_images)
    imags_temp = np.ndarray((N_images, img_rows, img_cols, 3), dtype=np.uint8)
    for i in range(N_images):
            imag_original = imageio.imread(os.path.join(image_path, filenames[i] + '.jpg'))
            imag_original = np.uint8(imag_original)
            imag = resize(imag_original, (img_rows, img_cols, 3), order = 0)
            imag = np.uint8(imag*255.)
            imags_temp[i] = imag

            pbar.update(1)
    pbar.close()
    imags = imags_temp
    
    return imags, filenames


# %%
print()
print('-'*30, 'Loading and preprocessing test data...','-'*30)
imags_test, filenames = get_data_test(image_path, img_rows, img_cols)
imags_test = np.float32(imags_test)
imags_test /= 255.

print()
print('-'*30, 'Loading model...','-'*30)
model = load_model(os.path.join(model_dir, 'model_for_LG_staining_analysis.h5'))

print()
print('-'*30, 'Predicting masks on test data...','-'*30)
masks_test = model.predict(imags_test, batch_size=1, verbose=1)


# %%
N_images = imags_test.shape[0]

imags = np.uint8(imags_test*255.)

masks = np.float32(np.squeeze(masks_test, axis=3))
masks = (np.float32(masks>0.25) + np.float32(masks>0.75)) / 2.
masks = np.uint8(masks * 255.)


# %%
print()
print('-'*30, 'Calculating and Saving analysis values to a Excel file','-'*30)
print('Calculating...')

wb = Workbook()
ws1 = wb.active
ws1.title = "Result Values"
caption_array = [
    'Image Name', 
    'Ratio of white, %', 'Area of white, mm2', 
    'Ratio of stain, %', 'Area of stain, mm2', 
    'Percentage of staining, %']
ws1.append(caption_array)

N_totalpixels = img_rows * img_cols
pbar = tqdm(total = N_images)
for i in range(N_images):
    #imsave(os.path.join(pred_dir, filenames[i] + '_mask_predicted.png'), imgs_mask_test[i])

    N_roipixels  = np.float32(np.sum(masks[i]>100))
    N_maskpixels = np.float32(np.sum(masks[i]>200))

    Ratio_roi  = N_roipixels  / N_totalpixels * 100
    Area_roi   = area_image * Ratio_roi / 100
    Ratio_mask = N_maskpixels / N_totalpixels * 100
    Area_mask  = area_image * Ratio_mask / 100
    Ratio      = N_maskpixels / N_roipixels   * 100

    value_array = [
        filenames[i], 
        Ratio_roi, Area_roi, 
        Ratio_mask, Area_mask, 
        Ratio]  
    ws1.append(value_array)

    pbar.update(1)
pbar.close()

wb.save(filename = out_filename)
print('Saved into file "', out_filename, '"')


# %%
print()
print('-'*30, 'Saving preprocessed images to png files...','-'*30)
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
pbar = tqdm(total = N_images)
for i in range(N_images):
    imsave(os.path.join(pred_dir,  filenames[i] + '.png'), imags[i])
    pbar.update(1)
pbar.close()

print()
print('-'*30, 'Saving predicted masks to png files...','-'*30)
pbar = tqdm(total = N_images)
for i in range(N_images):
    imsave(os.path.join(pred_dir, filenames[i] + '_mask_predicted.png'), masks[i])
    pbar.update(1)
pbar.close()

print('-'*30, 'Saving analysed images to png files...','-'*30)
pbar = tqdm(total = N_images)
for i in range(N_images):
    data = np.array(imags[i])
    white_areas = (masks[i] > 200)
    data[white_areas, :] = (255,63,192)
    imsave(os.path.join(pred_dir,  filenames[i] + '_analysed.png'), data)
    pbar.update(1)
pbar.close()

print('Analysis Completed!')