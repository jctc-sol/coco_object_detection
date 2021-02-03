import torch.nn.functional as F
import torchvision.transforms.functional as FT
from functools import partial
from torch import nn
from torchvision.models import vgg16
from dataset import CocoDataset
from utils   import *
from math import sqrt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.
    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    
    Code: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())
    return tensor


class VGGBase(nn.Module):
    """
    Base VGG module of SSD network
    Code: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py
    """
    def __init__(self):
        super(VGGBase, self).__init__()
        # standard convolutional layers in VGG16
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1   = nn.MaxPool2d(kernel_size=2, stride=2)
        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims
        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)
        # replacements for FC6 & FC7
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.load_params()
        
        
    def load_params(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        # pretrained VGG base
        vgg = vgg16(True)
        pretrained_state_dict = vgg.state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # transfer conv. parameters from pretrained model to current model
        for i, param in enumerate([p for p in param_names if 'features' in p]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)
        self.load_state_dict(state_dict)

    
    def forward(self, image):
        # conv1 fwd sequence
        out = F.relu(self.conv1_1(image))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)
        # conv2 fwd sequence
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)
        # conv3 fwd sequence
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)
        # conv4 fwd sequence
        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        conv4_3_features = out
        out = self.pool4(out)
        # conv5 fwd sequence
        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)
        # conv6
        out = F.relu(self.conv6(out))
        conv7_features = F.relu(self.conv7(out))
        return conv4_3_features, conv7_features
    
    
class AuxLayers(nn.Module):
    """
    Auxiliary layers subsequent to the VGG base module of SSD
    Code: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py
    """
    def __init__(self):
        super(AuxLayers, self).__init__()
        # Conv8_2 layer components
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1
        # Conv9_2 layer components
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1
        # Conv10_2 layer components
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0
        # Conv11_2 layer components
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0
        # init layer parameters
        self.init_conv2d()
        

    def init_conv2d(self):
        """
        Initialize convolution parameters with Xavier uniform
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
                
                
    def forward(self, conv7_features):
        # conv8 fwd sequences
        out = F.relu(self.conv8_1(conv7_features))
        out = F.relu(self.conv8_2(out))
        conv8_2_ft = out
        # conv9 fwd sequences
        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_2_ft = out
        # conv10 fwd sequences
        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        conv10_2_ft = out
        # conv11 fwd sequences
        out = F.relu(self.conv11_1(out))
        out = F.relu(self.conv11_2(out))
        conv11_2_ft = out
        return conv8_2_ft, conv9_2_ft, conv10_2_ft, conv11_2_ft
    
    
class PredLayers(nn.Module):
    """
    Prediction conv layers to output bound box output and class probabilities
    Code: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py
    """
    def __init__(self, n_classes):
        super(PredLayers, self).__init__()
        self.n_classes = n_classes
        
        # Define how many bounding boxes (with different aspect ratio)
        # there to be per grid location
        n_boxes = {'conv4_3' : 4,
                   'conv7'   : 6,
                   'conv8_2' : 6,
                   'conv9_2' : 6,
                   'conv10_2': 4,
                   'conv11_2': 4}
        
        # Bounding box offset predictors
        self.loc_conv4_3  = nn.Conv2d(512 , n_boxes['conv4_3']*4, kernel_size=3, padding=1)
        self.loc_conv7    = nn.Conv2d(1024, n_boxes['conv7']*4,   kernel_size=3, padding=1)
        self.loc_conv8_2  = nn.Conv2d(512 , n_boxes['conv8_2']*4, kernel_size=3, padding=1)
        self.loc_conv9_2  = nn.Conv2d(256 , n_boxes['conv9_2']*4,  kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256 , n_boxes['conv10_2']*4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256 , n_boxes['conv11_2']*4, kernel_size=3, padding=1)
        
        # Object class predictors
        self.cl_conv4_3  = nn.Conv2d(512,  n_boxes['conv4_3'] * n_classes,  kernel_size=3, padding=1)
        self.cl_conv7    = nn.Conv2d(1024, n_boxes['conv7'] * n_classes,    kernel_size=3, padding=1)
        self.cl_conv8_2  = nn.Conv2d(512,  n_boxes['conv8_2'] * n_classes,  kernel_size=3, padding=1)
        self.cl_conv9_2  = nn.Conv2d(256,  n_boxes['conv9_2'] * n_classes,  kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256,  n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256,  n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)
        
        # Initalize all convolution parameters
        self.init_conv2d()
        
    
    def init_conv2d(self):
        """
        Init conv2d layers with Xavier norm and 0 bias
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
                
                
    def forward(self, conv4_3_ft, conv7_ft, conv8_2_ft, conv9_2_ft, conv10_2_ft, conv11_2_ft):
        batch_size = conv4_3_ft.size(0)

        # Locator outputs for bounding box
        # --------------------------------
        # Conv4_3 locator
        l_conv4_3 = self.loc_conv4_3(conv4_3_ft)             # (N, 16, 38, 38)
        # note: contiguous() ensures tensor is stored in a contiguous 
        # chunk of memory; needed for calling .view() for reshaping below
        l_conv4_3 = l_conv4_3.permute(0,2,3,1).contiguous()  # (N, 38, 38, 16)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)        # (N, 5776, 4), total of 5776 bound boxes         
        # Conv7 locator
        l_conv7 = self.loc_conv7(conv7_ft)                   # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0,2,3,1).contiguous()      # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)            # (N, 2166, 4)        
        # Conv8 locator
        l_conv8_2 = self.loc_conv8_2(conv8_2_ft)             # (N, 24, 19, 19)
        l_conv8_2 = l_conv8_2.permute(0,2,3,1).contiguous()  # (N, 19, 19, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)        # (N, 2166, 4)         
        # Conv9 locator
        l_conv9_2 = self.loc_conv9_2(conv9_2_ft)             # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0,2,3,1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)        # (N, 150, 4)        
        # Conv10 locator
        l_conv10_2 = self.loc_conv10_2(conv10_2_ft)            # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0,2,3,1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)        # (N, 150, 4)        
        # Conv11 locator
        l_conv11_2 = self.loc_conv11_2(conv11_2_ft)            # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0,2,3,1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)        # (N, 4, 4)
        
        # Class prediction outputs for each bounding box
        # ----------------------------------------------
        # Conv4_3 classifier
        cl_conv4_3 = self.cl_conv4_3(conv4_3_ft)                       # (N, 4 boxes * n_classes, 38, 38)
        cl_conv4_3 = cl_conv4_3.permute(0,2,3,1).contiguous()          # (N, 38, 38, 4 boxes * n_classes)
        cl_conv4_3 = cl_conv4_3.view(batch_size, -1, self.n_classes)   # (N, 5776, n_classes)
        # Conv7 classifier
        cl_conv7   = self.cl_conv7(conv7_ft)                           # (N, 6 boxes * n_classes, 19, 19)
        cl_conv7   = cl_conv7.permute(0,2,3,1).contiguous()            # (N, 19, 19, 6 boxes * n_classes)
        cl_conv7   = cl_conv7.view(batch_size, -1, self.n_classes)     # (N, 2166, n_classes)
        # Conv8_2 classifier
        cl_conv8_2 = self.cl_conv8_2(conv8_2_ft)                       # (N, 6 boxes * n_classes, 10, 10)
        cl_conv8_2 = cl_conv8_2.permute(0,2,3,1).contiguous()          # (N, 10, 10, 6 boxes * n_classes)
        cl_conv8_2 = cl_conv8_2.view(batch_size, -1, self.n_classes)   # (N, 600, n_classes)
        # Conv9_2 classifier
        cl_conv9_2 = self.cl_conv9_2(conv9_2_ft)                       # (N, 6 boxes * n_classes, 5, 5)
        cl_conv9_2 = cl_conv9_2.permute(0,2,3,1).contiguous()          # (N, 5, 5, 6 boxes * n_classes)
        cl_conv9_2 = cl_conv9_2.view(batch_size, -1, self.n_classes)   # (N, 150, n_classes)
        # Conv10_2 classifier
        cl_conv10_2 = self.cl_conv10_2(conv10_2_ft)                    # (N, 4 boxes * n_classes, 3, 3)
        cl_conv10_2 = cl_conv10_2.permute(0,2,3,1).contiguous()        # (N, 3, 3, 4 boxes * n_classes)
        cl_conv10_2 = cl_conv10_2.view(batch_size, -1, self.n_classes) # (N, 36, n_classes)
        # Conv11_2 classifier
        cl_conv11_2 = self.cl_conv11_2(conv11_2_ft)                    # (N, 4 boxes * n_classes, 1, 1)
        cl_conv11_2 = cl_conv11_2.permute(0,2,3,1).contiguous()        # (N, 1, 1, 4 boxes * n_classes)
        cl_conv11_2 = cl_conv11_2.view(batch_size, -1, self.n_classes) # (N, 4, n_classes)  
        
        # Concatenate all locators and all classifiers
        # There are a total of 5776 + 2166 + 600 + 150 + 36 + 4 = 8732 bounding box locations in total
        locations = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        class_scores = torch.cat([cl_conv4_3, cl_conv7, cl_conv8_2, cl_conv9_2, cl_conv10_2, cl_conv11_2], dim=1)
        
        return locations, class_scores
    
    
class SSD300(nn.Module):
    
    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        self.n_classes = n_classes
        # network components
        self.base = VGGBase()
        self.aux  = AuxLayers()
        self.pred = PredLayers(self.n_classes)
        # rescale factor 
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20) # init values to 20
        # create prior boxes
        self.prior_boxes = self.create_prior_boxes()
        
        
    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        # size of kernels in each respective feature maps
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}
        
        # relative scale of each feature map to the input image
        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        # different aspect ratio bounding boxes to use at each feature map layer
        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}
        
        fmaps = list(fmap_dims.keys())
        prior_boxes = []

        # iterate through each feature map
        for k, fmap in enumerate(fmaps):
            
            # go through each grid-location on the feature map (i, j)
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    
                    # compute bounding box center coordinates normalized against the size of feature map dimension
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]
                    
                    # populate bounding boxes of different aspect ratio to prior_boxes list
                    for ratio in aspect_ratios[fmap]:
                        # bounding boxes defined in terms [center_x_coord, center_y_coord, w, h]
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map (i.e. index out of bound in fmaps[k+1]) 
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # shape (8732, 4)
        prior_boxes.clamp_(0, 1) # truncate all values between [0,1]

        return prior_boxes
    
    

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                                         conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores