import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools import mask


def drawSample(sample):
    # visualize the sample
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
        draw.rectangle([x0,y0,x1,y1], outline='orange')
#         # go through all segmaps & draw
#         segmap = sample['segs'][i][0]
#         draw.polygon(segmap, fill=(0, 255, 255, 125))
    plt.imshow(img); plt.axis('off')


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