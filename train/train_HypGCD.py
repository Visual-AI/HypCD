import argparse
import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import AverageMeter
from project_utils.general_utils import init_experiment, get_mean_lr, str2bool
from project_utils.cluster_and_log_utils import log_accs_from_preds
from models import vision_transformer as vits1
from models import vision_transformer2 as vits2

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from config import exp_root, dino_pretrain_path, dinov2_pretrain_path

import warnings
from loguru import logger
warnings.filterwarnings("ignore", category=DeprecationWarning)

# hyperbolic
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, hyper=False, c=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.hyper = hyper
        self.c = c

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

        device = (torch.device('cuda')
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
        if self.hyper:
            similarity_matrix = -dist_matrix(anchor_feature, contrast_feature, c=self.c)
        else:
            similarity_matrix = torch.matmul(F.normalize(anchor_feature, dim=-1, p=2), F.normalize(contrast_feature, dim=-1, p=2).T)

        anchor_dot_contrast = torch.div(similarity_matrix, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, args, hyper):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    if hyper:
        similarity_matrix = -dist_matrix(features, features, c=args.c)
    else:
        similarity_matrix = torch.matmul(F.normalize(features, dim=-1, p=2), F.normalize(features, dim=-1, p=2).T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def train(projection_head, model, train_loader, test_loader, unlabelled_train_loader, args):

    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)
    sup_con_crit_distance = SupConLoss(hyper=True, c=args.c)
    sup_con_crit_angle = SupConLoss(hyper=False, c=args.c)
    best_test_acc_lab = 0

    for epoch in range(args.epochs):

        loss_record = AverageMeter()
        projection_head.train()
        model.train()

        start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            data_time = time.perf_counter() - start

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = torch.cat(images, dim=0).to(device)

            # Extract features with base model
            features = model(images)
            pstr = ''

            # Pass features through projection head
            features = projection_head(features)

            loss_angle, loss_distance = 0.0, 0.0
            # Choose which instances to run the contrastive loss on
            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                # Contrastive loss for all examples
                con_feats = features
            # distance unsup loss
            contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=args, hyper=True)
            contrastive_loss_distance = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
            loss_distance += (1 - args.sup_con_weight) * contrastive_loss_distance
            pstr += f'distance contrastive_loss: {contrastive_loss_distance.item():.4f} '

            # angle unsup loss
            contrastive_logits_angle, contrastive_labels_angle = info_nce_logits(features=con_feats, args=args, hyper=False)
            contrastive_loss_angle = torch.nn.CrossEntropyLoss()(contrastive_logits_angle, contrastive_labels_angle)
            loss_angle += (1 - args.sup_con_weight) * contrastive_loss_angle
            pstr += f'angle contrastive_loss: {contrastive_loss_angle.item():.4f} '

            # Supervised contrastive loss
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]

            # distance sup loss
            sup_con_loss_distance = sup_con_crit_distance(sup_con_feats, labels=sup_con_labels)
            loss_distance += args.sup_con_weight * sup_con_loss_distance
            pstr += f'distance sup_con_loss: {sup_con_loss_distance.item():.4f} '

            # angle sup loss
            sup_con_loss_angle = sup_con_crit_angle(sup_con_feats, labels=sup_con_labels)
            loss_angle += args.sup_con_weight * sup_con_loss_angle
            pstr += f'angle sup_con_loss: {sup_con_loss_angle.item():.4f} '

            # combine distance and angle loss
            lambda_distance = (epoch - (args.hyper_start_epoch - 1)) / ((args.hyper_end_epoch - 1) - (args.hyper_start_epoch - 1))
            lambda_distance = torch.max(torch.tensor([0, lambda_distance])).item()
            lambda_distance = torch.min(torch.tensor([1, lambda_distance])).item()
            lambda_distance = lambda_distance * args.hyper_max_weight
            loss = (1 - lambda_distance) * loss_angle + lambda_distance * loss_distance

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            whole_time = time.perf_counter() - start
            start = time.perf_counter()
            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t time {:.3f} data_time {:.3f} loss {:.3f}\t {}'.format(epoch, batch_idx, len(train_loader), whole_time, data_time, loss.item(), pstr))


        # args.logger.info('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc Euc: {:.4f} | Seen Class Acc Hyp: {:.4f}'.format(epoch, loss_record.avg, train_acc_record_euc.avg, train_acc_record_hyp.avg))

        # Step schedule
        exp_lr_scheduler.step()

        if epoch:
            with torch.no_grad():
                args.logger.info('Testing on unlabelled examples in the training data...')
                all_acc, old_acc, new_acc = test_kmeans(model, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
                args.logger.info('Testing on disjoint test set...')
                all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader, epoch=epoch, save_name='Test ACC', args=args)
            # ----------------
            # LOG
            # ----------------
            args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

            torch.save(model.state_dict(), args.model_path)
            args.logger.info("model saved to {}.".format(args.model_path))

            torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')
            args.logger.info("projection head saved to {}.".format(args.model_path[:-3] + '_proj_head.pt'))

            if old_acc_test > best_test_acc_lab:
                best_test_acc_lab = old_acc_test

                args.logger.info(f'Best ACC on Old Classes on disjoint test set: {old_acc_test:.4f}...')
                args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

                torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
                args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
                args.logger.info("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best.pt'))





def test_kmeans(model, test_loader, epoch, save_name, args):

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    args.logger.info('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):

        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    args.logger.info('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    args.logger.info('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, writer=args.writer)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dino', type=str, default='v1')

    # hyperbolic
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_model_path', default=None, type=str)
    parser.add_argument('--hyper_start_epoch', default=0, type=int)
    parser.add_argument('--hyper_end_epoch', default=200, type=int)
    parser.add_argument('--c', type=float, default=0.1)
    parser.add_argument('--cr', type=float, default=2.3)
    parser.add_argument('--riemannian', type=bool, default=False)
    parser.add_argument('--hyp_dim', default=256, type=int)
    parser.add_argument('--hyper_max_weight', type=float, default=1.0)
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    set_random_seed(args.seed)
    args.logger = logger
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[f'HypGCD_{args.dataset_name}'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    # Add a handler for stdout and configure it to log to stdout as well
    args.logger.add(sys.stdout)

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path

        # DINO version
        if args.dino == 'v1':
            model = vits1.__dict__['vit_base']()
        elif args.dino == 'v2':
            model = vits2.__dict__['vit_base']()
            pretrain_path = dinov2_pretrain_path
        else:
            raise AttributeError('Unsupported DINO version')

        model.load_state_dict(torch.load(pretrain_path, map_location='cpu'))

        if args.warmup_model_dir is not None:
            args.logger.info(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

        model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = 256

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
    else:
        raise NotImplementedError

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    mlp = vits1.__dict__['DINOHead2'](in_dim=args.feat_dim, bottleneck_dim=args.hyp_dim)

    # hyperbolic projection
    last = hypnn.ToPoincare(c=args.c, ball_dim=args.hyp_dim, riemannian=args.riemannian, clip_r=args.cr)
    projection_head = nn.Sequential(mlp, last)
    projection_head.to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    if args.eval_only:
        if args.eval_model_path is not None:
            print(f'Loading evaluation model weights from {args.eval_model_path}')
            model.load_state_dict(torch.load(args.eval_model_path, map_location='cpu'))
            with torch.no_grad():
                all_acc, old_acc, new_acc = test_kmeans(model, test_loader_unlabelled, epoch=0, save_name='Train ACC Unlabelled', args=args)
                args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
                all_acc, old_acc, new_acc = test_kmeans(model, test_loader_unlabelled, epoch=0, save_name='Train ACC Unlabelled', args=args)
                args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
    else:
        train(projection_head, model, train_loader, test_loader_labelled, test_loader_unlabelled, args)