# %%
import os
import imageio
import datetime
import numpy as np

from tqdm import tqdm
from openpyxl import Workbook

from skimage.io import imsave
from skimage.transform import resize
from skimage.color import rgb2lab

from tensorflow.keras.models import load_model

# %%
image_path = './test_images/'
model_dir = './models'

current_time = datetime.datetime.now() 
time_str = current_time.strftime("%Y%m%d_%H%M%S")
pred_dir = './Eye_Redness_Analysis_Result__' + time_str
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
out_filepath = os.path.join(pred_dir,'Eye_Redness_Analysis_Result_Values_' + time_str + '.xlsx')

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
model = load_model(os.path.join(model_dir, 'model_eyeball_seg.h5'))

print()
print('-'*30, 'Predicting masks on test data...','-'*30)
masks_test = model.predict(imags_test, batch_size=1, verbose=1)


# %%
N_images = imags_test.shape[0]

imags = np.uint8(imags_test*255.)

masks = np.float32(np.squeeze(masks_test, axis=3))
masks = np.float32(masks>0.5)
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
    'Ratio of red, %', 'Area of red, mm2', 
    'Percentage of red, %']
ws1.append(caption_array)

N_totalpixels = img_rows * img_cols
pbar = tqdm(total = N_images)
for i in range(N_images):
    imag = imags[i]
    eyew = masks[i]
    
    Lab = rgb2lab(imag)
    Lab0 = [100000, 104.5514, 0.6981]
    Lab0[0] = np.mean(Lab[:,:,0])

    dLab = np.zeros((Lab.shape[0], Lab.shape[1]))
    for c in range(Lab.shape[-1]):
        d = Lab[:, :, c] - Lab0[c]
        dLab += np.square(d)
    dLab = np.sqrt(dLab)
    
    red_area = (dLab < 90)
    red_area[eyew<=200]=0
    
    N_eyewpixels = np.float32(np.sum(eyew>200))
    N_redpixels = np.float32(np.sum(red_area))

    Ratio_eyew  = N_eyewpixels  / N_totalpixels * 100
    Area_eyew   = area_image * Ratio_eyew / 100
    
    Ratio_red = N_redpixels / N_totalpixels * 100
    Area_red  = area_image * Ratio_red / 100
    Ratio      = N_redpixels / N_eyewpixels   * 100

    value_array = [
        filenames[i], 
        Ratio_eyew, Area_eyew, 
        Ratio_red, Area_red, 
        Ratio]  
    ws1.append(value_array)


    imsave(os.path.join(pred_dir,  filenames[i] + '.png'), imag)
    imsave(os.path.join(pred_dir, filenames[i] + '_mask_predicted.png'), eyew)

    data = np.array(imag)
    data[red_area, :] = (0,205,205)
    
    imsave(os.path.join(pred_dir,  filenames[i] + '_analysed.png'), data)
    




    pbar.update(1)
pbar.close()

wb.save(filename = out_filepath)
print('Saved into file "', out_filepath, '"')


print('Analysis Completed!')