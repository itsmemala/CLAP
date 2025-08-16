"""
Implementation taken from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
Contains custom extension of the SupCon Loss
"""
from __future__ import print_function

import torch
import torch.nn as nn
import math
import sys


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07,
                 use_supcon_pos=False,
                 num_samples=None,
                 bs=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.use_supcon_pos = use_supcon_pos
        self.num_samples = num_samples
        if self.num_samples is not None:
            self.prompt_mask = torch.zeros_like(torch.empty(bs, bs))
            for k in range(bs):
                prompt_idx = math.floor(k/self.num_samples) # get prompt idx for this sample
                start_idx = prompt_idx*self.num_samples
                end_idx = start_idx + self.num_samples
                self.prompt_mask[k][start_idx:end_idx] = 1

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (features.get_device() #torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # this does nothing when contrast_count=1
        # mask-out self-contrast cases
        logits_mask = torch.scatter( # this is 1 everywhere except diagonal
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask # this sets the diagonal elements to 0
        # print(mask)
        # sys.exit()
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        if self.num_samples is not None:
            all_prompt_mask, wp_mask, wp_samples = [], [], []
            # for k in range(len(mask)):
            #     prompt_mask = torch.zeros_like(mask[k])
            #     prompt_idx = math.floor(k/self.num_samples) # get prompt idx for this sample
            #     start_idx = prompt_idx*self.num_samples
            #     end_idx = start_idx + self.num_samples
            #     prompt_mask[start_idx:end_idx] = 1
            #     sample_wp_mask = mask[k].detach().clone()*prompt_mask # keep only samples from same prompt
            #     if sample_wp_mask.sum()==0: continue # skip samples with no within-prompt positive pairs
            #     wp_samples.append(k)
            #     wp_mask.append(sample_wp_mask)
            #     all_prompt_mask.append(prompt_mask)
            # orig_mask, all_prompt_mask = torch.stack(wp_mask), torch.stack(all_prompt_mask)
            # orig_log_prob = logits[wp_samples] - torch.log((all_prompt_mask*exp_logits[wp_samples]).sum(1, keepdim=True)) # Use all_prompt_mask to normalise over only pairs from same prompt
            
            if mask.shape[0]!=self.prompt_mask.shape[0]:
                self.prompt_mask = torch.zeros_like(mask)
                for k in range(mask.shape[0]):
                    prompt_idx = math.floor(k/self.num_samples) # get prompt idx for this sample
                    start_idx = prompt_idx*self.num_samples
                    end_idx = start_idx + self.num_samples
                    self.prompt_mask[k][start_idx:end_idx] = 1
            self.prompt_mask = self.prompt_mask.to(device)
            wp_mask = mask.detach().clone()*self.prompt_mask # keep only samples from same prompt
            wp_samples = torch.argwhere(wp_mask.sum(dim=1)>0).squeeze() # skip samples with no within-prompt positive pairs
            mask = wp_mask[wp_samples]
            log_prob = logits[wp_samples] - torch.log((self.prompt_mask[wp_samples]*exp_logits[wp_samples]).sum(1, keepdim=True)) # Use self.prompt_mask to normalise over only pairs from same prompt
            # try:
            #     assert orig_log_prob.sum() == log_prob.sum()
            # except AssertionError:
            #     print(orig_log_prob.sum(),log_prob.sum())

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1) # this gives the number of positives for each sample
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs # this computes the loss for each sample as the average over all positive pairs for that sample
        if self.use_supcon_pos: mean_log_prob_pos = mean_log_prob_pos[(torch.squeeze(labels)==1).nonzero()] # select only positive class samples (i.e we do not want to pull together negative class samples)
                
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.view(anchor_count, mean_log_prob_pos.shape[0]).mean()
        # sys.exit()

        return loss