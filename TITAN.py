from pickletools import optimize
from time import time
from tkinter.messagebox import NO
from traceback import print_tb
from matplotlib import widgets
from torch import autograd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from loguru import logger
import torch.nn.functional as F
from model_loader import load_model
from evaluate import AccuracyCalculator, mutil_task_eval
import random
from PIL import ImageFilter
import torch
from torch import nn
import numpy as np
from pylab import mpl
import matplotlib.pyplot as plt


class BaseClassificationLoss(nn.Module):
    def __init__(self):
        super(BaseClassificationLoss, self).__init__()
        self.losses = {}

    def forward(self, logits, code_logits, labels, onehot=True):
        raise NotImplementedError

def get_imbalance_mask(sigmoid_logits, labels, nclass, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / nclass

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale
    mask[labels == 1] = 1
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()
    return mask, multiclass_acc


class StdHashLoss(BaseClassificationLoss):
    def __init__(self,
                 ce=1,
                 s=8, 
                 m=0.4, 
                 m_type='cos', 
                 multiclass=False, 
                 quan=0, 
                 quan_type='cs',
                 multiclass_loss='bce',
                 **kwargs):
        super(StdHashLoss, self).__init__()
        self.ce = ce
        self.s = s
        self.m = m
        self.m_type = m_type
        self.multiclass = multiclass
        self.quan = quan
        self.quan_type = quan_type
        self.multiclass_loss = multiclass_loss
        assert multiclass_loss in ['bce', 'imbalance', 'label_smoothing']

    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def forward(self, logits, labels, code_logits = None, onehot=False):
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            margin_logits = self.compute_margin_logits(logits, labels)

            if self.multiclass_loss in ['bce', 'imbalance']:
                loss_ce = F.binary_cross_entropy_with_logits(margin_logits, labels, reduction='none')
                if self.multiclass_loss == 'imbalance':
                    imbalance_mask, multiclass_acc = get_imbalance_mask(torch.sigmoid(margin_logits), labels,
                                                                        labels.size(1))
                    loss_ce = loss_ce * imbalance_mask
                    loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
                    self.losses['multiclass_acc'] = multiclass_acc
                else:
                    loss_ce = loss_ce.mean()
            elif self.multiclass_loss in ['label_smoothing']:
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1)
                loss_ce = loss_ce.mean()
            else:
                raise NotImplementedError(f'unknown method: {self.multiclass_loss}')
        
        else:
            
            if onehot:
                labels = labels.argmax(1)
            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)
            loss_ce_batch = F.cross_entropy(margin_logits, labels, reduction='none')

        self.losses['ce'] = loss_ce
        loss = self.ce * loss_ce 
        return loss

#----------------------optimizer-------------------------------
def build_optimizer(args, model):
    optimizer = getattr(torch.optim, 'Adam')(
        list(model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    return optimizer

#-------------------eval-----------------------
def feat_extractor(model, data_loader, logger=None, is_np=True):
    model.eval()
    # numpy
    if is_np == True:
        feats = list()
        for i, batch in enumerate(data_loader):
            imgs = batch[0].cuda()  
            with torch.no_grad():
                out = model(imgs)[0].data.cpu().numpy() # h
                feats.append(out)
            del out
        feats = np.vstack(feats)
    # tensor
    else:
        feats = list()
        for i, batch in enumerate(data_loader):
            imgs = batch[0].cuda()  
            with torch.no_grad():
                out = model(imgs)[0]
                feats.append(out)
            del out
        feats = torch.vstack(feats)
    model.train()
    return feats

#----------------------------------prototype---------------------
def construct_distribution(model, data_loader, args):
    with torch.no_grad():
        prototype = torch.zeros(args.num_class, args.in_channels).cuda()
        prototype_var = torch.zeros(args.num_class, args.in_channels).cuda()
        features = []
        labels = []
        Index = []
        for batch in data_loader:
            images = batch[0].cuda()
            targets = torch.argmax(batch[1], dim = 1) 
            index = batch[2]
            _, feats = model(images)
            features.append(feats)
            labels.append(targets)
            Index.append(index)
        features = torch.vstack(features)
        labels = torch.concat(labels)
        Index = torch.concat(Index)
        # distribution
        for cls in range(args.num_class):
            mask = torch.where(labels == cls)[0]
            if mask.shape[0] != 0:
                f = features[mask]
                prototype[cls,:] = torch.mean(f, dim = 0)
                if mask.shape[0] > 1: # fix bug
                    prototype_var[cls,:] = torch.std(f, dim = 0)
    return prototype, prototype_var, features, labels, Index


def train(  
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            args
    ):
    max_iter = args.max_iter
    evaluate_interval = args.evaluate_interval
    topK = args.topk
    device = args.device
    best_mapk = 0
    best_precise = 0
    best_recall = 0
    Start_training_time = time()
    model = load_model(args)   
    model.to(device)
    K_max = 20
    optimizer = build_optimizer(args, model) 
    criterion = StdHashLoss()
    prototype = torch.zeros(args.num_class, args.in_channels).cuda()
    prototype_var = torch.zeros(args.num_class, args.in_channels).cuda()
    ema = 0.9

    # classification warm up
    for warmup in range(args.warm_up):
        for (images, targets_onehot, index) in train_dataloader:
            images = images.to(device)
            targets = torch.argmax(targets_onehot, dim = 1)
            optimizer.zero_grad()
            targets = targets.to(device)    
            h, feats = model(images) 
            logits = model.head.logits(h)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
    
    
    for epoch in range(max_iter):

        prototype_epoch, prototype_var_epoch, feats, labels, Index = construct_distribution(model, train_dataloader, args)
        if epoch == 0:
            prototype, prototype_var = prototype_epoch, prototype_var_epoch
        else:
            prototype, prototype_var = ema * prototype + (1 - ema) * prototype_epoch, ema * prototype_var + (1 - ema) * prototype_var_epoch

        # warm up then refine
        Knn = (epoch / (max_iter * 2)) * K_max
        if Knn >= 1:
            for i in range(prototype.shape[0]):
                centerids = prototype[i]
                topk_id = torch.argsort(torch.sum((centerids - feats)**2,dim=1))[:int(Knn)]
                for candidate in topk_id:
                    if labels[candidate] == i:
                        continue
                    else:
                        rebuttal = torch.argsort(torch.sum((prototype[labels[candidate]] - feats)**2,dim=1))[:int(Knn)]
                        if candidate in rebuttal:
                            continue
                        else:
                            labels[candidate] = i

        for (images, targets_onehot, index) in train_dataloader:
            images = images.to(device) 
            Pos = torch.where(torch.eq(Index.unsqueeze(0), index.unsqueeze(1)))[1]
            targets = labels[Pos]
            targets = targets.to(device)    
            h, feats = model(images) 
            m = torch.distributions.beta.Beta(torch.tensor([2.]), torch.tensor([2.]))
            lamda = m.sample().to(device)
            optimizer.zero_grad()

            ids = torch.argmax(feats @ prototype.T, dim = 1).cuda()
            inx = torch.arange(ids.shape[0])
            weights = (feats @ prototype.T)[inx,ids] + 1 
            mixup_features = lamda * feats + (1 - lamda) * (prototype[ids] + 0.05 * prototype_var[ids])    
            mix_feature = model.head.fc(mixup_features)

            prototype_pred = feats @ prototype.T
            mix_prototype_pred = mixup_features @ prototype.T

            confidence_score = []
            for i in range(targets.shape[0]):
                sim_exp = torch.exp(prototype_pred[i]) 
                softmax_loss = sim_exp[targets[i]] / torch.sum(sim_exp)
                confidence_score.append(softmax_loss)
            confidence_score = torch.tensor(confidence_score)

            clean_mask = torch.argsort(confidence_score)[int(len(confidence_score) * args.noise_rate):]
            clean_pred = prototype_pred[clean_mask]
            clean_targets = targets[clean_mask]
            loss1 = criterion(clean_pred, clean_targets)

            pos_mask = targets.expand(targets.shape[0], targets.shape[0]).t() == targets.expand(targets.shape[0], targets.shape[0])
            sim_mat = torch.matmul(mix_feature, mix_feature.t()) # mixup is work
            sim_mat = (sim_mat.T * weights).T
            neg_mask = (~pos_mask) & (sim_mat > 0.5)  # hard negative 
            pos_mask = pos_mask & (sim_mat < (1 - 1e-5))
            pos_pair = (sim_mat)[pos_mask]
            neg_pair = (sim_mat)[neg_mask]
            pos_loss = torch.sum(-pos_pair + 1)
            neg_loss = torch.sum(neg_pair)
            loss2 = (pos_loss + neg_loss) / targets.shape[0]

            loss = loss1 + loss2
            loss.backward()
            optimizer.step() 

        logger.info('[Epoch:{}/{}][loss_all:{:.4f}][Memory_used:{} GB]'.format(epoch+1, max_iter, loss.item(), torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0))
        
        if epoch % evaluate_interval == evaluate_interval - 1:
            if args.tag == "CARS196" or args.tag == "CUB200" or args.tag == 'CIFAR10' or args.tag == 'Cars98N':
                ret_metric = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r", "r_precision",'mean_average_precision_at_100'), exclude=())
                labels_query = query_dataloader.dataset.get_targets()
                labels_gallery = retrieval_dataloader.dataset.get_targets()
                feats_query = feat_extractor(model,query_dataloader,logger)
                feats_gallery = feat_extractor(model,retrieval_dataloader,logger)
                ret_metric =  ret_metric.get_accuracy(feats_query, feats_gallery, labels_query, labels_gallery, True) 
                mAP = ret_metric['mean_average_precision_at_r']
                precise = ret_metric['precision_at_1']
                recall = ret_metric['r_precision']

                # onehot_query_targets = query_dataloader.dataset.get_targets_onehot()
                # onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets_onehot()
                # np.save("code/{}/query_code_{}_mAP_{}".format(args.tag, args.method , mAP), feats_query)
                # np.save("code/{}/retrieval_code_{}_mAP_{}".format(args.tag, args.method , mAP), feats_gallery)
                # np.save("code/{}/query_target_{}_mAP_{}".format(args.tag, args.method , mAP), onehot_query_targets.cpu().detach().numpy())
                # np.save("code/{}/retrieval_target_{}_mAP_{}".format(args.tag, args.method , mAP), onehot_retrieval_targets.cpu().detach().numpy())
                
                logger.info('[iter:{}/{}][mAP@R:{:.4f}][Precise@1:{:.4f}][R_Precise:{:.4f}]'.format(
                        epoch+1,
                        max_iter,
                        mAP,
                        precise,
                        recall
                    ))

                best_mapk = max(best_mapk, mAP)
                best_precise = max(best_precise, precise)
                best_recall = max(best_recall, recall)

            elif args.tag == "FLICKR25K":  
                labels_query = torch.tensor(query_dataloader.dataset.get_targets())
                labels_gallery = torch.tensor(retrieval_dataloader.dataset.get_targets()) 
                feats_query = feat_extractor(model,query_dataloader,logger,False)
                feats_gallery = feat_extractor(model,retrieval_dataloader,logger,False)
                mAP, precise = mutil_task_eval(feats_query, feats_gallery, labels_query, labels_gallery, device, topK)
                onehot_query_targets = query_dataloader.dataset.get_targets_onehot()
                onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets_onehot()

                # np.save("code/{}/query_code_{}_mAP_{}".format(args.tag, args.method , mAP), feats_query.cpu())
                # np.save("code/{}/retrieval_code_{}_mAP_{}".format(args.tag, args.method , mAP), feats_gallery.cpu())
                # np.save("code/{}/query_target_{}_mAP_{}".format(args.tag, args.method , mAP), onehot_query_targets.cpu().detach().numpy())
                # np.save("code/{}/retrieval_target_{}_mAP_{}".format(args.tag, args.method , mAP), onehot_retrieval_targets.cpu().detach().numpy())
                
                logger.info('[iter:{}/{}][mAP@R:{:.4f}][Precise@1:{:.4f}]'.format(
                        epoch+1,
                        max_iter,
                        mAP,
                        precise
                    ))
                best_mapk = max(best_mapk, mAP)
                best_precise = max(best_precise, precise)
            else:
                raise ValueError("No sucn bencnmark")
            
            # torch.save({'iteration': epoch,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # }, os.path.join('checkpoints','resume{}.t'.format(epoch)))
            
    End_training_time = time()
    total_time = (End_training_time - Start_training_time) / 60 
    logger.info('[Train Finish:{} mins][Best mAP@R:{:.4f}][Best Precise@1:{:.4f}][Best R_Precise:{:.4f}]'.format(
            total_time, 
            best_mapk,
            best_precise,
            best_recall
        ))
            
    


