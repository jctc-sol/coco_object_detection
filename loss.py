from utils import *
from torch import nn


class MultiBoxLoss(nn.Module):
    """
    Loss funcion for object detection, which is a linear combination of:
    a) object localization loss for the predicted bounding box location; and
    b) classification loss for the predicted object class
    
    Code reference: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py#L532
    """
    
    def __init__(self, pboxes, threshold=0.5, neg_pos_ratio=3, alpha=1., device=None):
        """
        :param pboxes: prior bounding boxes of object detection model, provided in center coordinates
        :param threshold: cutoff threshold on IoU overlap between a pair of true object box and prior bounding box
        :param neg_pos_ratio: ratio to be used in hard negative sample minning
        :param alpha: relative weighting between localization & classification losses
        """
        super(MultiBoxLoss, self).__init__()
        # setup device
        if device:
            if type(device)==torch.device: 
                self.device=device
            elif type(device)==str:
                self.device=torch.device(device)
        else:
            # defaults to cuda:0 if cuda is available
            if torch.cuda.is_available(): self.device = torch.device('cuda:0') 
            else: self.device = torch.device('cpu')
        # coordinate transforms
        self.boundaryCoord = BoundaryCoord()
        self.offsetCoord   = OffsetCoord()
        # convert prior boxes from center coordinates to boundary coordinates
        self.pboxes_cc = pboxes.to(self.device)
        self.pboxes_bc = self.boundaryCoord.encode(pboxes).to(self.device)
        # params
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        # localization/classification losses
        self.loc_loss = nn.L1Loss()
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        
        
    def forward(self, pred_boxes, pred_scores, true_boxes, true_classes):
        """
        Forward pass to compute the loss given predicted bounding boxes and predicted classification scores
        from an object detection model. N for batch size below.
        :param pred_boxes:  predicted bound boxes from object detection model in offset coordinates form; tensor of dim (N, 8732, 4)
        :param pred_scores: predicted classification scores from boject detection model; tensor of dim (N, 8732, n_classes)
        :param true_boxes: ground truth label on location of each object in a batch of images in boundary coordinates; List object of N tensors
        :param true_classes: grounth truth label on class of each object in a batch of images; List object of N tensors
        
        :return: scalar loss measure
        """        
        bs = pred_boxes.size(0)
        n_classes = pred_scores.size(-1)
        n_priors  = self.pboxes_cc.size(0)
        assert n_priors == pred_boxes.size(1) == pred_scores.size(1)
                
        # ---------------------------------------------------------------------------------
        # 1. iterate over each image in the batch to populate true_locs and true_cls
        # ---------------------------------------------------------------------------------        
        true_locs = torch.zeros_like(pred_boxes, dtype=torch.float).to(self.device)  # (N, 8732, 4)
        true_cls  = torch.zeros((bs, n_priors),  dtype=torch.long).to(self.device)   # (N, 8732)
        
        for i in range(bs):
            # get # of ground truth objects in the image
            n_objs = true_boxes[i].size(0)

            # ---------------------------------------------------------------------------------
            # i. compute IoU overlaps between object boxes and prior boxes
            # ---------------------------------------------------------------------------------
            # need true boxes in both center-coordinate and boundary-coordinate forms
            true_boxes_bc = true_boxes[i]
            true_boxes_cc = self.boundaryCoord.decode(true_boxes[i])

            # find overlap of each ground truth objects with each of the prior bboxes
            # (note: that both set of boxes need to be in boundary-coordinate format)
            overlaps = find_jaccard_overlap(true_boxes_bc, self.pboxes_bc)  # (n_objects, 8732)

            # ---------------------------------------------------------------------------------
            # ii. find best pbox for each object based on best IoU & suppress background objects
            # ---------------------------------------------------------------------------------
            # for each prior boxes, find the ground truth object that overlaps the most with it
            obj_overlap_with_prior, obj_assigned_to_pbox = overlaps.max(dim=0)  # (8732)
            # conversely, for each obkect, find the prior boxes that ooverlaps the most with it
            _, best_pbox_for_obj = overlaps.max(dim=1)  # (n_objects)

            # ---------------------------------------------------------------------------------
            # iii. fix potential issues due to poor overlaping between object & prior boxes
            # ---------------------------------------------------------------------------------
            # one potential problem is poor overlap between object and any prior boxes
            # in this case, make sure each object is assign to the prior boxes with most overlap
            obj_assigned_to_pbox[best_pbox_for_obj] = torch.LongTensor(range(n_objs)).to(self.device)
            # get object class label for each prior bounding box
            obj_cls_for_pbox = true_classes[i][obj_assigned_to_pbox]
            
            # also ensure each object is captured by at least one prior box; manually change 
            # the overlap to 1 to avoid it getting assigned to background object after thresholding
            obj_overlap_with_prior[best_pbox_for_obj] = 1.
            obj_cls_for_pbox[obj_overlap_with_prior < self.threshold] = 0

            # ---------------------------------------------------------------------------------
            # iv. record the ground truths for this image
            # ---------------------------------------------------------------------------------
            # compute the offset coordinates of object locations relative to prior boxes
            true_locs[i] = self.offsetCoord.encode(true_boxes_cc[obj_assigned_to_pbox], self.pboxes_cc)
            # add true object class label allocation for each prior bounding box
            true_cls[i] = obj_cls_for_pbox          
        
        # ---------------------------------------------------------------------------------
        # 2. LOCALIZATION LOSS of non-background objects
        # ---------------------------------------------------------------------------------
        # get flag for all non-background prior bounding boxes (i.e. class label > 0)
        nonbackground_priors = true_cls != 0  # bit map of size (N, 8732)
        loc_loss = self.loc_loss(pred_boxes[nonbackground_priors], true_locs[nonbackground_priors])
        
        # ---------------------------------------------------------------------------------
        # 3. CLASSIFICATION LOSS of non-background objects
        # ---------------------------------------------------------------------------------
        cls_loss_all = self.cls_loss(pred_scores.view(-1, n_classes), true_cls.view(-1))
        cls_loss_all = cls_loss_all.view(bs, n_priors)  # reshape back to size (N, 8732)
        cls_loss_pos = cls_loss_all[nonbackground_priors].sum()

        # ---------------------------------------------------------------------------------
        # 4. Hard Negative Mining
        # ---------------------------------------------------------------------------------
        # HNM is used in the case where there is a large imbalance between -v vs +ve class.
        # This is often the case in object detection since the majority of bounding boxes 
        # would capture background objects (class = 0). Artificially balance out the 
        # -ve vs +ve class ratio by selecting `n` number of -ve samples with the largest loss 
        # (i.e. hardest -v samples)
        if self.neg_pos_ratio > 0:        
            # get number of hard negatives to sample
            n_positives   = nonbackground_priors.sum(dim=1).sum().float()
            n_neg_samples = self.neg_pos_ratio * n_positives

            # set positive prior losses to 0 since we've already computed cls_loss_pos
            cls_loss_neg = cls_loss_all.clone()
            cls_loss_neg[nonbackground_priors] = 0.
            # sort losses in decending order
            cls_loss_neg, _ = cls_loss_neg.sort(dim=1, descending=True)
            hardness_ranks  = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(cls_loss_neg).to(self.device)  # (N, 8732)
            # get the hard -ve samples and sum of their losses
            hard_neg = hardness_ranks < n_neg_samples.unsqueeze(-1) # (N, 8732)
            cls_loss_hard_neg = cls_loss_neg[hard_neg].sum()
        
            # COMBINED CLASSIFICATION LOSS
            # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
            cls_loss = (cls_loss_pos + cls_loss_hard_neg) / n_positives
        else:
            cls_loss = cls_loss_pos

        # total loss
        return loc_loss + self.alpha * cls_loss
