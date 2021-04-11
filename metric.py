import torch.nn.functional as F
import torchvision.transforms.functional as FT
from functools import partial
from torch import nn
from utils   import *


class mAP():
    
    def __init__(self, n_classes, device):
        """
        :param n_classes: number of class objects to compute mean AP over; note that 0 
                          should be reserved as background class
        """
        self.name = "mAP"
        self.n_classes = n_classes
        if device is None:
            self.device = "cpu"
        else:
            self.device = device
    
        
    def concat_batch_tensors(self, boxes, labels, scores=None):
        """
        Each batch contains M images, and each image contains N_i objects within (since each object contains
        different numbr of objects). As a result, each of boxes, labels, scores are in the form of list of 
        tensors. This helper function simply concatenates all tensors within the list into a single one for
        the batch.
        
        :param boxes: list of M tensors each of size (N_i, 4) for bounding boxes of each image within the batch
        :param labels: list of M tensors each of size (N_i, self.n_class) for labels of objects within each image within the batch
        :param scores: list of M tensors each of size (N_i, self.n_class) for confidence scores for each object 
                       within each image within the batch
                       
        :return img_idx: 1-D tensor with size being the total number of objects in the image batch, each 
                         entry tells which image the object belongs to
        :return boxes: 2-D tensor with size (n_total_objects_in_batch, 4)
        :return labels: 1-D tensor with size (n_total_objects_in_batch)
        :return scores: 1-D tensor with size (n_total_objects_in_batch)
        """
        
        # initialize a list to keep track of the image corresponding to entries in the list
        img_idx, n_images = list(), len(labels)
        for idx in range(n_images):
            n_objects_in_img = boxes[idx].size(0)
            img_idx.extend([idx] * n_objects_in_img)        
        img_idx = torch.LongTensor(img_idx).to(self.device)
        boxes   = torch.cat(boxes, dim=0)
        labels  = torch.cat(labels, dim=0)
        assert img_idx.size(0) == boxes.size(0) == labels.size(0), "tensor size mismatch"
        if scores is not None:
            scores = torch.cat(scores, dim=0)
        return {'img'   : img_idx, 
                'boxes' : boxes, 
                'labels': labels, 
                'scores': scores}
        
        
    def class_specific_mAP(self, truths, preds, category):
        """
        :param truths: dictionary containing ground truth information with keys 'img', 'boxes', 'labels', 'scores'
        :param preds:  dictionary containing predicted information with keys 'img', 'boxes', 'labels', 'scores'
        :param category: integer representing the category of interest
        
        : return mAP_class: mean average precision of detections related to category
        """
        # get predictions related to this category
        pred_labels = preds['labels']
        pred_class_images = preds['img'][pred_labels==category]
        pred_class_boxes  = preds['boxes'][pred_labels==category]
        pred_class_scores = preds['scores'][pred_labels==category]
        n_detections      = pred_class_boxes.size(0)
        # mAP is simply 0 if there's nothing detected to be in this class
        if n_detections == 0:
            return 0.0

        # get ground truths related to this category
        true_labels = truths['labels']
        true_class_images = truths['img'][true_labels==category]
        true_class_boxes  = truths['boxes'][true_labels==category]

        # re-order scores/images/boxes by descending confidence score
        pred_class_scores, sort_idx = torch.sort(pred_class_scores, dim=0, descending=True)
        pred_class_images = pred_class_images[sort_idx]
        pred_class_boxes  = pred_class_boxes[sort_idx]
        
        # initialize tensors to keep track of:
        # a) which true objects with this class have been 'detected'
        true_class_boxes_detected = torch.zeros((true_class_boxes.size(0)), dtype=torch.uint8).to(self.device)
        # b) which detected boxes are true positives
        tp = torch.zeros((n_detections), dtype=torch.float).to(self.device)
        # c) which detected boxes are flase positives
        fp = torch.zeros((n_detections), dtype=torch.float).to(self.device)
        
        # iterate through each detection & check whether it is true-positive or false-positive
        for d in range(n_detections):
            # get the image this detection is made on + bounding box & score associated with this detection
            this_img   = pred_class_images[d]
            this_box   = pred_class_boxes[d].unsqueeze(0)
            this_score = pred_class_scores[d].unsqueeze(0)
            # get ground truth boxes for this image
            true_boxes = true_class_boxes[true_class_images==this_img]
            # if there are no boxes in this image matching this category, then mark as false-positive
            if true_boxes.size(0) == 0:
                fp[d] = 1
                continue
                
            # compute Jaccard overlaps; if there are significant level of overlap regions between the 
            # current (single) detected bounding box and ground truth boxes (multiple), then it is a 
            # true-positive; false-positives otherwise
            overlaps = find_jaccard_overlap(this_box, true_boxes) # (1, n_true_objects_in_img)
            max_overlap, idx = torch.max(overlaps.squeeze(0), dim=0)
            # get the original_idx position of this object within the true_class_boxes_detected tensor
            # this is used to check whether this object has already been detected prior
            origin_idx = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images==this_img][idx]            
            # if max overlap is greater than 0.5 threshold, this prediction has detected this object
            if max_overlap.item() > 0.5:
                # check whether this object has been detected before
                if true_class_boxes_detected[origin_idx] == 0:
                    tp[d] = 1
                else:
                    fp[d] = 1
        
        # consolidate how many true-positive & true-positive detections there were
        cumsum_tp = torch.cumsum(tp, dim=0)  # (n_class_detections) cumulative sums
        cumsum_fp = torch.cumsum(fp, dim=0)  # (n_class_detections) cumulative sums
        cumsum_precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
        cumsum_recall    = cumsum_tp / true_class_boxes.size(0)  # note: we ignored difficulties
        
        # create thresholds between [0,1] with 0.1 increments
        recall_thresholds = torch.arange(start=0, end=1.1, step=0.1).tolist()
        precisions        = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(self.device)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumsum_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumsum_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        class_mAP = precisions.mean().item()        
        return class_mAP
                            
        
    def __call__(self, pred_boxes, pred_labels, pred_scores, true_boxes, true_labels):
        """
        Takes in both prediction & ground truth labeling for both bounding boxes and object class labels
        to compute the mean average precision (mAP). This function operates on a batch of images at a 
        time, and because each image contain different number of objects within, each input should be 
        provided as a list of tensors (where each entry of the list is for one particular image).
        
        For each of the below, M represents the `batch_size`, `n_i` is the number of predicted objects in 
        each image, and `N_i` is the number of true objects in each image.
        
        :param pred_boxes: predicted bounding boxes for each image, list of M tensors each of size (n_i, 4)
        :param pred_labels: predicted class label for each image, list of M tensors each of size (n_i, self.n_class)
        :param pred_scores: predicted class score for each image, list of M tensors each of size (n_i, self.n_class)
        :param true_boxes: ground truths bounding boxes, list of M tensors each of size (N_i, 4)
        :param true_labels: ground truths class label for each iamge, list of M tensors each of size (N_i, self.n_class)
        
        :return: list of average precisions for all classes, mean average precision (mAP)
        """
        # check length of list of each input is consistent
        assert len(pred_boxes) == len(pred_labels) == len(pred_scores) == len(true_boxes) == len(true_labels),\
        "input tensor length mismatch"
        
        # we want to concatenate the list of tensors together within each list, to do that, we first need 
        # to track which object belongs to which image
        truths = self.concat_batch_tensors(true_boxes, true_labels)
        
        # because the number of predicted objects may not necessarily match the actual number of objects, we 
        # also need to do the same for predicted tensors separately
        preds  = self.concat_batch_tensors(pred_boxes, pred_labels, pred_scores)
        
        # iterate over each category to compute the average precision of the detections for that category 
        avg_precisions = torch.zeros((self.n_classes - 1), dtype=torch.float)
        AP = {}
        for c in range(1, self.n_classes):
            avg_precisions[c-1] = self.class_specific_mAP(truths, preds, c)
            AP[c] = avg_precisions[c-1]
        # further compute the mean over the average precisions over all categories
        mAP = avg_precisions.mean().item()
        
        return mAP, AP
