# %% 
# Required modules
import time
import numpy as np 
import matplotlib.pyplot as plt

import scipy.ndimage
import scipy.misc
import random

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure, img_as_float

# %%
def rotate(images, labels, theta = None):
    # Rotate volume by a minor angle (+/- 10 degrees: determined by investigation of dataset variability)
    if theta is None:
        theta = random.randint(-30, 30)
    img_new = scipy.ndimage.interpolation.rotate(images, theta, reshape = False)
    lbl_new = scipy.ndimage.interpolation.rotate(labels, theta, reshape = False)
    return img_new, lbl_new

def scale_and_crop(images, labels):
    scale_factor = random.uniform(1, 1.5)

    channs = images.shape[2]
    images_zoom = images
    labels_zoom = labels
    for c in range(channs):
        # Scale the volume by a minor size and crop around centre (can also modify for random crop)
        image = images[:,:,c]
        label = labels
        o_s = image.shape
        r_s = [0]*len(o_s)
        img_zoom = scipy.ndimage.interpolation.zoom(image, scale_factor, order=0)
        new_shape = img_zoom.shape
        # Start with offset
        for i in range(len(o_s)):
            if new_shape[i] == 1: 
                r_s[i] = 0
                continue
            r_c = int(((new_shape[i] - o_s[i]) - 1)/2)
            r_s[i] = r_c
        r_e = [r_s[i] + o_s[i] for i in list(range(len(o_s)))]
        images_zoom[:,:,c] = img_zoom[  r_s[0]:r_e[0], 
                                        r_s[1]:r_e[1]   ]
        
    lbl_zoom = scipy.ndimage.interpolation.zoom(label, scale_factor, order=0)
    labels_zoom[:,:] = lbl_zoom[    r_s[0]:r_e[0], 
                                    r_s[1]:r_e[1]   ]
    return images_zoom, labels_zoom

def grayscale_variation(images, labels):
    # Introduce a random global increment in gray-level value of volume. 
    im_min = np.min(images)
    im_max = np.max(images)
    mean = np.random.normal(0, 0.1)
    smp = np.random.normal(mean, 0.01, size = np.shape(images))
    images = images + im_max*smp
    images[images <= im_min] = im_min # Clamp to min value
    images[images > im_max] = im_max  # Clamp to max value
    return images, labels

def elastic_deformation(images, labels, 
                        alpha=None, sigma=None, 
                        mode="constant", cval=0, is_random=False): 
    # Apply elastic deformation/distortion to the volume
    if alpha == None:
        alpha=images.shape[1]*3.
    if sigma == None:
        sigma=images.shape[1]*0.07
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))
        
    shape = (images.shape[0], images.shape[1])
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=mode, cval=cval) * alpha
    x_, y_ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
    
    new_images = np.zeros(images.shape)
    new_labels = np.zeros(labels.shape)
    for i in range(images.shape[2]): # apply the same distortion to each slice within the volume
        new_images[:,:,i] = map_coordinates(images[:,:,i], indices, order=0).reshape(shape)
    for i in range(labels.shape[2]): # apply the same distortion to each slice within the volume
        new_labels[:,:,i] = map_coordinates(labels[:,:,i], indices, order=0).reshape(shape)
        
    return new_images, new_labels

def sample_with_p(p):
    # Helper function to return boolean of a sample with given probability p
    if random.random() < p:
        return True
    else:
        return False

def get_random_perturbation(images, labels):
    # Generate a random perturbation of the input feature + label
    p_rotate = 0.9
    p_scale = 0.9
    p_gray = 0.9
    p_deform = 0.9
    new_images, new_labels = images, labels
    if sample_with_p(p_rotate):
        new_images, new_labels = rotate(new_images, new_labels)
    if sample_with_p(p_scale):
        new_images, new_labels = scale_and_crop(new_images, new_labels)
    #if sample_with_p(p_gray):
    #    new_images, new_labels = grayscale_variation(new_images, new_labels)
    if sample_with_p(p_deform):
        new_images, new_labels = elastic_deformation(new_images, new_labels)
    return new_images, new_labels
