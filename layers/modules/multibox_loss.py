import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, log_sum_exp

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_pred, conf_pred, size_tr_pred, ori_pred, priors = predictions
        num = loc_pred.size(0) # batch size
        priors = priors[:loc_pred.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        conf_t = torch.LongTensor(num, num_priors)

        loc_t = torch.Tensor(num, num_priors, 4)
        size_tr_t = torch.Tensor(num, num_priors, 6) #(x, y, z, h, w, l)
        ori_t = torch.Tensor(num, num_priors, 2) #(a, ry)
        '''
        iterating through image in a batch
        '''
        for idx in range(num):
            loc_truths = targets[idx][:, :4].data # gt_location
            labels = targets[idx][:, -1].data  # gt_classes
            # TODO
            size_tr_truths = targets[idx][:, 8:14].data
            ori_truths = targets[idx][:, 14:16].data
            defaults = priors.data # from torch.autograd.Variable to torch.Tensor
            match(self.threshold, defaults, self.variance, labels, loc_truths, size_tr_truths, ori_truths, conf_t, loc_t, size_tr_t, ori_t ,idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            size_tr_t = size_tr_t.cuda()
            ori_t = ori_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        size_tr_t = Variable(size_tr_t, requires_grad=False)
        ori_t = Variable(ori_t, requires_grad=False)

        pos = conf_t > 1
        num_pos = pos.sum(keepdim=True)

        #####################################
        # Size-Translation Loss (Smooth L1) #
        #####################################
        pos_mask = pos.unsqueeze(pos.dim()).expand_as(size_tr_pred)
        size_p = size_tr_pred[pos_mask].view(-1, 6)
        size_t = size_tr_t[pos_mask].view(-1, 6)
        loss_loc_size = F.smooth_l1_loss(size_p, size_t, size_average=False)
        
        ################################
        # Orientation Loss (Smooth L1) #
        ################################
        pos_mask = pos.unsqueeze(pos.dim()).expand_as(ori_pred)
        ori_p = ori_pred[pos_mask].view(-1, 2)
        ori_t = ori_t[pos_mask].view(-1, 2)
        loss_ori = F.smooth_l1_loss(ori_p, ori_t, size_average=False)
        
        #################################
        # Localization Loss (Smooth L1) #
        #################################
        pos_mask = pos.unsqueeze(pos.dim()).expand_as(loc_pred)
        loc_p = loc_pred[pos_mask].view(-1, 4)
        loc_t = loc_t[pos_mask].view(-1, 4)
        loss_pixloc = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # loss_l = loss_l1 + loss_l2 + loss_l3   ##############################
        loss_l = (loss_pixloc, loss_ori, loss_loc_size)
        
        ###################
        # Confidence Loss #
        ###################
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_pred.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.gt(0).long().view(-1, 1))

        # Hard Negative Mining
        not_neg = conf_t > 0 # to remove the don't care labeled object from loss
        loss_c[not_neg] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)

        # BOUNDARY CONDITION: NO POSITIVE EXAMPLE IN THE BATCH
        if (num_pos.data==0)[0][0]:
            num_pos.data += 3
            not_neg.data += 3
            loss_l = None#Variable(torch.Tensor(0)).cuda()
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=not_neg.size(1)-1) # any problem with the max=pos.size(1)-1
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_pred)
        neg_idx = neg.unsqueeze(2).expand_as(conf_pred)
        conf_p = conf_pred[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        # modify the label of car to 1
        targets_weighted[targets_weighted.gt(0)] = 1

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = num_pos.data.sum()
        loss_c /= N
        diff = None
        return loss_l, loss_c, N, diff


