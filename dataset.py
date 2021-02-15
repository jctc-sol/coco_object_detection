import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask


class CocoDataset(Dataset):
    """
    Torch Dataset style implementation for COCO dataset. Requires pycocotools APIs.
    Use `conda install -c conda-forge pycocotools` to install.
    """
    
    def __init__(self, data_dir, dataset, anno_type, imgIds=[], catIds=[], transforms=None):
        """
        Initializes the pycocotools API as `self.coco`.
        Params:
        `data_dir`  : COCO style dataset directory
        `dataset`   : label of dataset (i.e. train/valid/test)
        `anno_type` : type of task (i.e. captions, instances, panoptic, person_keypoints, stuff)
        `imgIds`    : list of image IDs to select; defaults is empty list to load all images
        `catIds`    : list of category IDs to select; default is empty list to load all categories
        `transforms`: list of transforms to apply on each sample drawn from the dataset; defaults to None
        """
        self.dataset   = dataset
        self.img_dir   = f"{data_dir}/{dataset}"
        self.anno_file = f"{data_dir}/annotations/{anno_type}_{dataset}.json"
        self.coco = COCO(self.anno_file)
        self.imgIds    = imgIds
        self.catIds    = catIds
        self.transforms= transforms
        imgIds = self.coco.getImgIds(imgIds=imgIds, catIds=catIds)
        imgIds = list(sorted(imgIds))
        self.imgs = self.coco.loadImgs(imgIds)
        # get all categories
        allCatIds = self.coco.getCatIds()
        self.allcats = self.coco.loadCats(allCatIds)
        # create various cat->id and id->cat lookup dicts
        # note that cocoIds skip some integers so we re-create another set of category IDs 
        # that are continuous and reserve 0 for background
        self.cocoId2id, self.id2cocoId = {0:0}, {0:0}
        self.id2cat, self.cat2id = {0:'background'}, {'background':0}
        for i, cat in enumerate(self.allcats, 1):
            self.id2cocoId[i] = cat['id']
            self.cocoId2id[cat['id']] = i
            self.id2cat[i] = cat['name']
            self.cat2id[cat['name']] = i
        
    
    def __repr__(self):
        return f"COCO {self.dataset}; annoFile: {self.anno_file}; imgIds={self.imgIds}; catIds={self.catIds}"

    
    def __len__(self):
        return len(self.imgs)
        
        
    def __getitem__(self, idx):
        # load image using PIL for better integration with native torch transforms
        img = Image.open(f"{self.img_dir}/{self.imgs[idx]['file_name']}")
        # load annotations associated with the image
        annIds = self.coco.getAnnIds(imgIds=self.imgs[idx]['id'])
        annotations = self.coco.loadAnns(annIds)
        # parse annotations
        segmaps  = list()
        cats     = list()
        boxes    = list()
        for anno in annotations:
            if anno['iscrowd']==0:
                segmaps.append(anno['segmentation'])
                cats.append(self.cocoId2id[anno['category_id']])
                boxes.append(anno['bbox'])
        sample = {'image': img, # PILImage
                  'segs' : segmaps, # list of INT of length N
                  'cats' : np.stack(cats, axis=0)  if len(cats)>0  else np.stack([0], axis=0), # [N, 1]
                  'boxes': np.stack(boxes, axis=0) if len(boxes)>0 else np.stack([[0,0,0,0]], axis=0), # [N,4]
                 }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


    @classmethod
    def collate_fn(cls, batch, img_resized=False):
        """
        custom collate function (to be passed to the DataLoader) for combining tensors of 
        different sizes into lists.
        """
        images = list()
        segs   = list()
        cats   = list()
        boxes  = list()
        for sample in batch:
            images.append(sample['image'].unsqueeze(0))
            segs.append(sample['segs'])
            cats.append(sample['cats'])
            boxes.append(sample['boxes'])
        
        # if images have already been resized to same shape, then combine them into a 
        # single 4-D tensor of (B, C, H, W)
        if img_resized:
            images = torch.cat(images, 0)
                    
        batch = {'images': images,
                 'segs': segs,
                 'cats': cats,
                 'boxes': boxes
                }
        return batch