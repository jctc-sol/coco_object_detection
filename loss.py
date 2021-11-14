from utils import *
from torch import nn


class MultiBoxLoss(nn.Module):
    """
    Loss funcion for object detection, which is a linear combination of:
    a) object localization loss for the predicted bounding box location; and
    b) classification loss for the predicted object class
    
    Code reference: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py#L532
    """
    
    def __init__(self, img_sz, pboxes, threshold=0.5, neg_pos_ratio=3, alpha=1., device=None):
        """
        :param img_sz: input image size into object detection model (assumed to be square)
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
        self.cocoCoord = Coco2CenterCoord(img_sz,img_sz)
        self.boundaryCoord = BoundaryCoord()
        self.offsetCoord = OffsetCoord()
        # convert pboxes from center coordinates to boundary coordinates
        self.pboxes_cc = pboxes.to(self.device)
        self.pboxes_bc = self.boundaryCoord.encode(pboxes).to(self.device)
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
        :param true_boxes: ground truth label on location of each object in a batch of images, expressed in Coco coordinates; list of N tensors
        :param true_classes: grounth truth label on class of each object in a batch of images; list of N tensors
        
        :return: scalar loss measure
        """
        bs = pred_boxes.size(0)
        n_classes = pred_scores.size(-1)
        n_priors  = self.pboxes_cc.size(0)
        assert n_priors == pred_boxes.size(1) == pred_scores.size(1)
        
        # init tensors for recording all ground truth objects/labels allocated to each prior bounding boxes
        true_locs = torch.zeros_like(pred_boxes, dtype=torch.float).to(self.device)  # (N, 8732, 4)
        true_cls  = torch.zeros((bs, n_priors), dtype=torch.long).to(self.device)    # (N, 8732)
        
        # for each image in batch, we want to find the best ground truth object that each prior bounding box 
        # captures in terms of maximum IoU overlap. More specifically, we want to:
        # a) assign an object class to each prior bounding box that reflect the object class each prior box best overlaps with;
        #    a cutoff threshold is applied to suppress prior bounding boxes to background if IoU falls below this threhsold
        # b) compute how "off" each prior bound box location coordinate is relative to the ground truth object it has the best
        #    overlap with (i.e. as offset coordinates)
        # and populate `true_cls` & `true_locs` so they captured all the class/location-offset assignment for all prior bounding 
        # boxes for each image in the batch
        for i in range(bs):            
            # get number of ground truth objects in image i
            n_objs = true_boxes[i].size(0)
            # convert true_boxes from Coco coordinates to center coordinates
            true_boxes_cc = self.cocoCoord.encode(true_boxes[i])
            true_boxes_bc = self.boundaryCoord.encode(true_boxes_cc)
            # find overlap of each ground truth objects with each of the prior bboxes
            # note that both set of boxes need to be in boundary coord format
            overlaps = find_jaccard_overlap(true_boxes_bc, self.pboxes_bc)  # (n_objects, 8732)
            # find the best ground truth object overlaping with each prior bounding boxes
            obj_overlap_for_each_prior, obj_idx_for_each_prior = overlaps.max(dim=0)  # (8732)
            # conversely, find the best bounding box overlap with each ground truth objects
            _, best_pbox_for_each_obj = overlaps.max(dim=1)  # (n_objects)
                        
            # ** two potential problem scenarios to mitigate:
            # 1) none of the prior bounding boxes have overlap with groundtruth object > 0.5 
            #    and therefore the object is taken as background
            # Solution: assign each object to the corresponding maximum-overlap-prior
            obj_idx_for_each_prior[best_pbox_for_each_obj] = torch.LongTensor(range(n_objs)).to(self.device)
            # 2) a groundtruth object is not found as the maximum overlapped object with 
            #    any of the prior bounding boxes
            # Solution: artificially set IoU overlap with the best bounding box to 1 to 
            #           ensure each object is captured by 1 prior bounding box
            obj_overlap_for_each_prior[best_pbox_for_each_obj] = 1.

            # get object class label for each prior bounding box
            obj_label_for_each_prior = true_classes[i][obj_idx_for_each_prior]
            # for those with overlap < threshold, suppress object as background (i.e. class_label=0)
            obj_label_for_each_prior[obj_overlap_for_each_prior < self.threshold] = 0
            
            # add true object class label allocation for each prior bounding box
            true_cls[i] = obj_label_for_each_prior            
            # convert the ground truth object locations into offset coordinate format wrt to prior bboxes
            # note: both true_boxes & self.pboxes need to be expressed in center coordinates            
            true_locs[i] = self.offsetCoord.encode(true_boxes_cc[obj_idx_for_each_prior], self.pboxes_cc)

        # create flag for all non-background prior bounding boxes (i.e. class label = 0)
        positive_priors = true_cls != 0
        # LOCALIZATION LOSS across non-background prior bounding boxes
        loc_loss = self.loc_loss(pred_boxes[positive_priors], true_locs[positive_priors])
        
        # compute classification loss for all prior bounding boxes
        cls_loss_all = self.cls_loss(pred_scores.view(-1, n_classes), true_cls.view(-1))
        cls_loss_all = cls_loss_all.view(bs, n_priors)  # (N, 8732)
        
        # POSITIVE PRIOR CLASSIFICATION LOSS
        # gather the classification loss for all the positive prior bounding boxes
        cls_loss_pos_priors = cls_loss_all[positive_priors].sum()
        
        # Hard-Negative-Mining (HNM)
        # HNM is used in the case where there is a large imbalance between negative vs positive class 
        # ground truth objects. In the context of object detection this is often the case as the vast 
        # majority of bounding boxes would capture background (i.e. class = 0). Thus we artificially
        # balance out the negative vs positive class ratio by selecting `n` number of negative samples
        # with the largest loss (i.e. hardest negative samples) and include those in our loss computation
        # along with the positive classes
        n_positives = positive_priors.sum(dim=1).sum().float()
        n_neg_samples = self.neg_pos_ratio * n_positives
        # set positive prior losses to 0 since we've already computed cls_loss_pos_priors
        cls_loss_neg = cls_loss_all.clone()
        cls_loss_neg[positive_priors] = 0.
        # sort losses in decending order
        cls_loss_neg, _ = cls_loss_all.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(cls_loss_neg).to(self.device)  # (N, 8732)        
        hard_neg = hardness_ranks < n_neg_samples.unsqueeze(-1) # (N, 8732)
        # HNM LOSS
        cls_loss_hard_neg = cls_loss_neg[hard_neg].sum()
        
        # COMBINED CLASSIFICATION LOSS
        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        cls_loss = (cls_loss_pos_priors + cls_loss_hard_neg) / n_positives
        
        return loc_loss + self.alpha * cls_loss