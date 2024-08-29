# if __name__ == '__main__':
import matplotlib
# Agg backend runs without a display
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import json
import numpy as np
import skimage.io
import skimage.draw
from skimage.transform import resize
from tqdm import tqdm

img_rows = 512
img_cols = 512
img_chns = 3


data_dir = 'datasets/Eye_Redness'

# We mostly care about the x and y coordinates of each region
# Note: In VIA 2.0, regions was changed from a dict to a list.
annotations = json.load(open(os.path.join(data_dir, "via_annotations.json")))
annotations = list(annotations.values())  # don't need the dict keys

# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.
annotations = [a for a in annotations if a['regions']]

print('Converting Json annotations to masks and Resizing images and masks...')
pbar = tqdm(total = len(annotations))
for anno in annotations:
    filename = anno['filename'].split('.')[0]
    image_path = os.path.join(data_dir, "images", anno['filename'])
    
    image = skimage.io.imread(image_path)
    imag = resize(image, (img_rows, img_cols, 3), order = 0)
    imag = np.uint8(imag*255.)

    imag_path = os.path.join(data_dir, "dataset", filename +'.png')
    skimage.io.imsave(imag_path, imag)
    

    if type(anno['regions']) is dict:
        polygons = [r['shape_attributes'] for r in anno['regions'].values()]
        classes = [r['region_attributes'] for r in anno['regions'].values()]
    else:
        polygons = [r['shape_attributes'] for r in anno['regions']] 
        classes = [r['region_attributes'] for r in anno['regions']] 

    height, width = image.shape[:2]
    image_info = {
            "id": anno['filename'],
            "path": image_path,
            'polygons': polygons,
            'height': height,
            'width': width
    }    

    # Convert polygons to a bitmap mask of shape
    info = image_info
    mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                    dtype=np.uint8)
    for i, p in enumerate(info["polygons"]):
        # Get indexes of pixels inside the polygon and set them to 1
        if p['name'] == 'polygon' or p['name'] == 'polyline':
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])            
        elif p['name'] == 'circle':
            rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
            # rr, cc = skimage.draw.disk((p['cy'], p['cx']), p['r'])
        elif p['name'] == 'ellipse': 
            rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'], rotation=np.deg2rad(p['theta']))  

        rr[ (rr > info['height']-1) ] = info['height']-1
        cc[ (cc > info['width']-1) ] = info['width']-1

        mask[rr, cc, i] = 255
        
        mask = resize(mask, (img_rows, img_cols), order = 0)
        mask = np.uint8(mask*255.)

        mask_path = os.path.join(data_dir, "dataset", filename +'_MASK.png')
        skimage.io.imsave(mask_path, mask)
        # skimage.io.imshow(mask)
        # plt.show()

    pbar.update(1)
pbar.close()




