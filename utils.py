import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools import mask


def drawSample(sample):
    """
    Visualize the given sample by showing the image and
    draws bounding boxes of target objects in the image
    """
    img  = sample['image']
    if isinstance(img, torch.Tensor):
        tf = transforms.ToPILImage()
        img = tf(img)
    draw = ImageDraw.Draw(img, 'RGBA')
    for i in range(sample['boxes'].shape[0]):
        # go through all boxes & draw
        # bbox coordinates are provides in terms of [x,y,width,height], and
        # box coordinates are measured from the top left image corner and are 0-indexed
        box = list(sample['boxes'][i,:])
        x0, y0 = box[0], box[1]
        x1, y1 = x0+box[2], y0+box[3]
        draw.rectangle([x0,y0,x1,y1], outline='orange', width=4)
#         # TODO: draw instance segmentation maps
#         # go through all segmaps & draw
#         segmap = sample['segs'][i][0]
#         draw.polygon(segmap, fill=(0, 255, 255, 125))
    plt.imshow(img); plt.axis('off')

    
def show_augmented_samples(ds, sample_idx=0, n=10):
    """
    Helper function to visualize the same image sample
    specified by `sample_idx`, `n` number of times; 
    each subjected to data augmentations applied to 
    dataset `ds`.
    """
    samples_per_row = 5
    num_rows = int(n/samples_per_row) if n%samples_per_row==0 else int(n/samples_per_row)+1
    plt.figure(figsize=(16, 2*num_rows))
    for i in range(n):
        plt.subplot(num_rows, samples_per_row, i+1)
        drawSample(ds[sample_idx])    


class PhotometricDistort(object):
    """
    (Ref: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py)
    Distort brightness, contrast, saturation, and hue, each with a probability of `p` chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    def __init__(self, p):
        self.proba = p
        self.distortions = [FT.adjust_brightness,
                            FT.adjust_contrast,
                            FT.adjust_saturation,
                            FT.adjust_hue]
        
    def __call__(self, sample):
        img = sample['image']
        random.shuffle(self.distortions)
        for d in self.distortions:
            if random.random() < self.proba:
                if d.__name__ == 'adjust_hue':
                    # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                    adjust_factor = random.uniform(-18 / 255., 18 / 255.)
                else:
                    # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                    adjust_factor = random.uniform(0.5, 1.5)
                # Apply this distortion
                sample['image'] = d(sample['image'], adjust_factor)
        return sample
    
    
class Zoomout(object):
    """
    (Ref: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py)
    Perform a zooming out operation by placing the image in a larger canvas of filler material.
    Helps to learn to detect smaller objects.
    """
    def __init__(self, p, max_scale=4):
        self.proba = p
        self.max_scale = max_scale
        
        
    def __call__(self, sample):
        image = sample['image']
        boxes = sample['boxes']        
        if random.random() < self.proba:            
            _, original_h, original_w = image.size()
            scale = random.uniform(1, self.max_scale)
            new_h = int(scale * original_h)
            new_w = int(scale * original_w)

            # Create such an image with the filler
            filler  = torch.FloatTensor([image[0,:,:].mean(), image[1,:,:].mean(), image[2,:,:].mean()])  # (3)
            new_img = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
            # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
            # because all expanded values will share the same memory, so changing one pixel will change all

            # Place the original image at random coordinates in this new image (origin at top-left of image)
            left = random.randint(0, new_w - original_w)
            right = left + original_w
            top = random.randint(0, new_h - original_h)
            bottom = top + original_h
            new_img[:, top:bottom, left:right] = image

            # Adjust bounding boxes' coordinates accordingly
            # (n_objects, 4), n_objects is the no. of objects in this image
            new_boxes = boxes + torch.FloatTensor([left, top, 0, 0]).unsqueeze(0)
            
            sample['image'] = new_img
            sample['boxes'] = new_boxes
        return sample
    
    
class Flip(object):
    """
    (Ref: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py)
    Perform a flip of the image about the vertical axis of the image.
    """
    def __init__(self, p):
        self.proba = p
        
        
    def __call__(self, sample):
        if random.random() < self.proba:
            image = sample['image']
            boxes = sample['boxes']
            # flip image
            sample['image'] = FT.hflip(image)
            # flip boxes
            boxes[:,0] = image.width - boxes[:,0] - 1 - boxes[:,2]
            sample['boxes'] = boxes
        return sample
    
    
class Resize(object):
    """
    Resize image and bounding boxes to specified target `size` in the 
    form of either integer or tuple (H, W)
    """
    def __init__(self, size):
        self.target_size = size
        

    def __call__(self, sample):
        # original image
        image = sample['image']
        height, width = image.size()[1], image.size()[2]
        # resize image
        new_image = FT.resize(image, self.target_size)
        new_height, new_width = new_image.size()[1], new_image.size()[2]
        # resize boxes
        boxes = sample['boxes']        
        boxes[:,0] = boxes[:,0] * new_width / width
        boxes[:,1] = boxes[:,1] * new_height / height
        boxes[:,2] = boxes[:,2] * new_width / width
        boxes[:,3] = boxes[:,3] * new_height / height
        sample['image'] = new_image
        sample['boxes'] = boxes
        return sample
    

class Normalize(object):
    """
    Normalizes the image tensor(s) (expected to be within the range [0,1]) 
    of dimensions [C x W x H] with specified `mean` and `std`. The default 
    values of `mean` and `std` are based on torchvision pretrained models
    as specified here: https://pytorch.org/docs/stable/torchvision/models.html
    """    
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std  = std
        
        
    def __call__(self, sample):
        image = sample['image']
        sample['image'] = FT.normalize(image, self.mean, self.std)
        return sample            


class ImageToTensor(object):
    """
    Provides encode/decode methods to transform PILimage to tensor and vice-versa.
    The forward call of the class method assumes the input is of type `dict` with 'image'
    as one of its keys that holds a corresponding value that is a PILimage, and performs
    `encode` on the image.
    """
    def __init__(self):
        self.encoder = transforms.ToTensor()
        self.decoder = transforms.ToPILImage()
        
    def __call__(self, sample):
        sample['image'] = self.encode(sample['image'])
        return sample

    def encode(self, img):
        return self.encoder(img)
    
    def decode(self, tensor):
        return self.decoder(tensor)
    
    
class CategoryToTensor(object):
    """
    Provides encodes/decodes methods to transform category classes from 
    np.array format to torch.tensor format and vice-versa. 
    The forward call of the class method assumes the input is of type `dict`
    with 'cats' as one of its keys that holds a corresponding value that is 
    a np.array, and performs `encode` on the numpy array.
    """        
    def __call__(self, sample):
        sample['cats'] = self.encode(sample['cats'])
        return sample

    def encode(self, cats):
        return torch.LongTensor(cats)
    
    def decode(self, tensor):
        return tensor.numpy()
    
    
class BoxToTensor(object):
    """
    Provides encodes/decodes methods to transform bounding boxes from 
    np.array format to torch.tensor format and vice-versa. 
    The forward call of the class method assumes the input is of type `dict`
    with 'boxes' as one of its keys that holds a corresponding value that is 
    a np.array, and performs `encode` on the numpy array.
    """       
    def __call__(self, sample):
        sample['boxes'] = self.encode(sample['boxes'])
        return sample

    def encode(self, boxes):
        return torch.FloatTensor(boxes)
    
    def decode(self, tensor):
        return tensor.numpy()