import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics import Accuracy

from utils.data import get_datamodule
from utils.nets import MultiHeadResNet
from utils.eval import ClusterMetrics
from utils.sinkhorn_knopp import SinkhornKnopp

import numpy as np
from argparse import ArgumentParser
from datetime import datetime

parser = ArgumentParser()

parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
parser.add_argument("--data_dir", default="/data/dataset/CIFAR100", type=str, help="data directory")
parser.add_argument("--download", default=False, action="store_true", help="whether to download")
parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
parser.add_argument("--num_base_classes", default=80, type=int, help="number of base classes")
parser.add_argument("--num_novel_classes", default=20, type=int, help="number of novel classes")

parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--base_lr", default=0.2, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")

parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=4, type=int, help="number of heads for clustering")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--num_views", default=2, type=int, help="number of views")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="NCD", type=str, help="wandb project")
parser.add_argument("--entity", default="ncd2022", type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")

parser.add_argument("--magic", default=False, action="store_true", help="use dim=1 in ce loss")

parser.add_argument("--batch_head", default=False, action="store_true", help="whether to use batch-wise experts")
parser.add_argument("--batch_head_multi_novel", default=False, action="store_true", help="whether to use multi heads in novel-batch expert")
parser.add_argument("--batch_head_reg", default=1.0, type=float, help="coefficient of regularization on batch-wise experts")
parser.add_argument("--alpha", default=1.0, type=float, help="prediction weight on batch-wise experts")

parser.add_argument("--queue_size", default=500, type=int, help="length of queue in memory")
parser.add_argument("--queue_alpha", default=0.5, type=float, help="target = alpha*online_target + (1-alpha)*queue_target")
parser.add_argument("--sharp", default=0.25, type=float, help="for sharpening the target distribution, lower is sharper")


class Discoverer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadResNet(
            arch=self.hparams.arch,
            low_res="CIFAR" in self.hparams.dataset,
            num_base=self.hparams.num_base_classes,
            num_novel=self.hparams.num_novel_classes,
            proj_dim=self.hparams.proj_dim,
            hidden_dim=self.hparams.hidden_dim,
            overcluster_factor=self.hparams.overcluster_factor,
            num_heads=self.hparams.num_heads,
            num_hidden_layers=self.hparams.num_hidden_layers,
            batch_head=self.hparams.batch_head,
            batch_head_multi_novel=self.hparams.batch_head_multi_novel
        )

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.epsilon_sk)

        # metrics
        self.metrics = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )
        self.metrics_inc = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )

        # buffer for best head tracking
        self.register_buffer("loss_per_head", torch.zeros(self.hparams.num_heads))
        if self.hparams.batch_head_multi_novel:
            self.register_buffer("loss_per_batch_head", torch.zeros(self.hparams.num_heads))

        # memory bank, only for novel samples
        if self.hparams.queue_size:
            self.register_buffer('queue_feats', torch.zeros(
                self.hparams.num_views, self.hparams.num_heads, self.hparams.queue_size, self.hparams.proj_dim))
            self.register_buffer('queue_feats_over', torch.zeros(
                self.hparams.num_views, self.hparams.num_heads, self.hparams.queue_size, self.hparams.proj_dim))
            self.register_buffer('queue_targets', torch.ones(
                self.hparams.num_views, self.hparams.num_heads, self.hparams.queue_size,
                self.hparams.num_novel_classes).mul_(-1))
            self.register_buffer('queue_targets_over', torch.ones(
                self.hparams.num_views, self.hparams.num_heads, self.hparams.queue_size,
                self.hparams.overcluster_factor * self.hparams.num_novel_classes).mul_(-1))
            self.register_buffer('queue_pointer', torch.zeros(1, dtype=torch.long))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        self.loss_per_head = torch.zeros_like(self.loss_per_head)
        if self.hparams.batch_head_multi_novel:
            self.loss_per_batch_head = torch.zeros_like(self.loss_per_batch_head)

    def unpack_batch(self, batch):
        if self.hparams.dataset == "ImageNet":
            views_lab, labels_lab, views_unlab, labels_unlab = batch
            views = [torch.cat([vl, vu]) for vl, vu in zip(views_lab, views_unlab)]
            labels = torch.cat([labels_lab, labels_unlab])
        else:
            views, labels = batch
        mask_base = labels < self.hparams.num_base_classes
        return views, labels, mask_base

    def cross_entropy_loss(self, preds, targets):
        if self.hparams.magic:
            # this is an interesting magic that boosts the performance, yet is wrong in implementation
            # it can be more interesting to investigate the reasons behind ;-)
            preds = F.log_softmax(preds / self.hparams.temperature, dim=1)
            return -torch.mean(torch.sum(targets * preds, dim=1))
        else:
            preds = F.log_softmax(preds / self.hparams.temperature, dim=-1)
            return -torch.mean(torch.sum(targets * preds, dim=-1))

    def swapped_prediction(self, logits, targets):
        loss = 0.0
        for view in range(self.hparams.num_views):
            for other_view in np.delete(range(self.hparams.num_views), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.hparams.num_views * (self.hparams.num_views - 1))

    def neighbor_targets(self, feats, queue_feat, queue_tar):
        sim = torch.einsum("vhbd, vhqd -> vhbq", feats, queue_feat)  # similarity between online feats and queue feats
        sim = F.softmax(sim / self.hparams.temperature, dim=-1)
        return torch.einsum("vhbq, vhqt -> vhbt", sim, queue_tar)

    def sharpen(self, prob):
        sharp_p = prob ** (1. / self.hparams.sharp)
        sharp_p /= torch.sum(sharp_p, dim=-1, keepdim=True)
        return sharp_p

    def normed(self, fea):
        return F.normalize(fea, dim=-1)

    @torch.no_grad()
    def queuing(self, feats, feats_over, targets, targets_over, in_size):
        pointer = int(self.queue_pointer)
        if (pointer + in_size) // self.hparams.queue_size == 0:
            self.queue_feats[:, :, pointer:pointer + in_size, :] = feats
            self.queue_targets[:, :, pointer:pointer + in_size, :] = targets
            self.queue_feats_over[:, :, pointer:pointer + in_size, :] = feats_over
            self.queue_targets_over[:, :, pointer:pointer + in_size, :] = targets_over
        else:
            new_point = (pointer + in_size) % self.hparams.queue_size
            self.queue_feats[:, :, pointer:, :] = feats[:, :, new_point:, :]
            self.queue_feats[:, :, :new_point, :] = feats[:, :, :new_point, :]
            self.queue_targets[:, :, pointer:, :] = targets[:, :, new_point:, :]
            self.queue_targets[:, :, :new_point, :] = targets[:, :, :new_point, :]
            self.queue_feats_over[:, :, pointer:, :] = feats_over[:, :, new_point:, :]
            self.queue_feats_over[:, :, :new_point, :] = feats_over[:, :, :new_point, :]
            self.queue_targets_over[:, :, pointer:, :] = targets_over[:, :, new_point:, :]
            self.queue_targets_over[:, :, :new_point, :] = targets_over[:, :, :new_point, :]
        self.queue_pointer[0] = (pointer + in_size) % self.hparams.queue_size

    def training_step(self, batch, _):
        views, labels, mask_base = self.unpack_batch(batch)
        nbc = self.hparams.num_base_classes
        nac = self.hparams.num_base_classes + self.hparams.num_novel_classes

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(views)

        # gather outputs and initialize targets
        # logits_base:       [view_num, bs, base_class_num]
        # logits_novel:      [view_num, head_num, bs, novel_class_num]
        # logits_novel_over: [view_num, head_num, bs, 3*novel_class_num]
        # logits:            [view_num, head_num, bs, base_class_num + novel_class_num]
        # logits_over:       [view_num, head_num, bs, base_class_num+3*novel_class_num]
        outputs["logits_base"] = outputs["logits_base"].unsqueeze(1).expand(-1, self.hparams.num_heads, -1, -1)

        logits = torch.cat([outputs["logits_base"], outputs["logits_novel"]], dim=-1)
        logits_over = torch.cat([outputs["logits_base"], outputs["logits_novel_over"]], dim=-1)

        targets = torch.zeros_like(logits)
        targets_over = torch.zeros_like(logits_over)

        if self.hparams.batch_head:
            outputs["logits_batch_base"] = outputs["logits_batch_base"].unsqueeze(1).expand(
                -1, self.hparams.num_heads, -1, -1)
            if not self.hparams.batch_head_multi_novel:
                outputs["logits_batch_novel"] = outputs["logits_batch_novel"].unsqueeze(1).expand(
                    -1, self.hparams.num_heads, -1, -1)
                outputs["logits_batch_novel_over"] = outputs["logits_batch_novel_over"].unsqueeze(1).expand(
                    -1, self.hparams.num_heads, -1, -1)

            logits_batch_base = outputs["logits_batch_base"][:, :, mask_base, :]
            logits_batch_novel = outputs["logits_batch_novel"][:, :, ~mask_base, :]
            logits_batch_novel_over = outputs["logits_batch_novel_over"][:, :, ~mask_base, :]

            targets_batch_base = torch.zeros_like(logits_batch_base)
            targets_batch_novel = torch.zeros_like(logits_batch_novel)
            targets_batch_novel_over = torch.zeros_like(logits_batch_novel_over)

            logits_batch = torch.zeros_like(outputs["logits_batch_base"])
            logits_batch_over = torch.zeros_like(outputs["logits_batch_novel_over"])
            logits_batch[:, :, mask_base, :] = logits_batch_base
            logits_batch[:, :, ~mask_base, :] = logits_batch_novel
            logits_batch_over[:, :, mask_base, :nac] = logits_batch_base
            logits_batch_over[:, :, ~mask_base, :] = logits_batch_novel_over

            targets_batch = torch.zeros_like(logits_batch)
            targets_batch_over = torch.zeros_like(logits_batch_over)

        # now create targets for base and novel samples
        # targets_base: [base_img_num, base_class_num]
        targets_base = F.one_hot(labels[mask_base], num_classes=self.hparams.num_base_classes).float().to(self.device)

        # generate pseudo-labels with sinkhorn-knopp and fill novel targets
        for v in range(self.hparams.num_views):
            for h in range(self.hparams.num_heads):
                targets[v, h, mask_base, :nbc] = targets_base.type_as(targets)
                targets_over[v, h, mask_base, :nbc] = targets_base.type_as(targets)
                targets[v, h, ~mask_base, nbc:] = self.sk(
                    outputs["logits_novel"][v, h, ~mask_base]).type_as(targets)
                targets_over[v, h, ~mask_base, nbc:] = self.sk(
                    outputs["logits_novel_over"][v, h, ~mask_base]).type_as(targets)
                if self.hparams.batch_head:
                    targets_batch_base[v, h, :, :nbc] = targets_base.type_as(targets)
                    targets_batch_novel[v, h, :, nbc:] = self.sk(
                        outputs["logits_batch_novel"][v, h, ~mask_base, nbc:]).type_as(targets)
                    targets_batch_novel_over[v, h, :, nbc:] = self.sk(
                        outputs["logits_batch_novel_over"][v, h, ~mask_base, nbc:]).type_as(targets)
                    targets_batch_novel[v, h, :, nbc:] = (
                        targets_batch_novel[v, h, :, nbc:] + targets[v, h, ~mask_base, nbc:]) / 2
                    targets_batch_novel_over[v, h, :, nbc:] = (
                        targets_batch_novel_over[v, h, :, nbc:] + targets_over[v, h, ~mask_base, nbc:]) / 2

                    targets[v, h, ~mask_base, nbc:] = targets_batch_novel[v, h, :, nbc:]
                    targets_over[v, h, ~mask_base, nbc:] = targets_batch_novel_over[v, h, :, nbc:]

                    targets_batch[v, h, mask_base, :] = targets_batch_base[v, h, :, :]
                    targets_batch[v, h, ~mask_base, :] = targets_batch_novel[v, h, :, :]
                    targets_batch_over[v, h, mask_base, :nac] = targets_batch_base[v, h, :, :]
                    targets_batch_over[v, h, ~mask_base, :] = targets_batch_novel_over[v, h, :, :]

        # now queue time
        if self.hparams.queue_size:
            if self.hparams.batch_head and self.hparams.batch_head_multi_novel:
                self.queuing(
                    self.normed(outputs["proj_feats_novel"][:, :, ~mask_base, :] +
                                outputs["proj_feats_batch_novel"][:, :, ~mask_base, :]),
                    self.normed(outputs["proj_feats_novel_over"][:, :, ~mask_base, :] +
                                outputs["proj_feats_batch_novel_over"][:, :, ~mask_base, :]),
                    targets[:, :, ~mask_base, nbc:],
                    targets_over[:, :, ~mask_base, nbc:],
                    int((~mask_base).sum())
                )
            else:
                self.queuing(
                    outputs["proj_feats_novel"][:, :, ~mask_base, :],
                    outputs["proj_feats_novel_over"][:, :, ~mask_base, :],
                    targets[:, :, ~mask_base, nbc:],
                    targets_over[:, :, ~mask_base, nbc:],
                    int((~mask_base).sum())
                )

            if -1 not in self.queue_targets:  # make sure the queue is full
                if self.hparams.batch_head and self.hparams.batch_head_multi_novel:
                    neighbor_tar = self.neighbor_targets(
                        self.normed(outputs["proj_feats_novel"][:, :, ~mask_base, :] +
                                    outputs["proj_feats_batch_novel"][:, :, ~mask_base, :]),
                        self.queue_feats.clone().detach(),
                        self.queue_targets.clone().detach()
                    )
                    neighbor_tar_over = self.neighbor_targets(
                        self.normed(outputs["proj_feats_novel_over"][:, :, ~mask_base, :] +
                                    outputs["proj_feats_batch_novel_over"][:, :, ~mask_base, :]),
                        self.queue_feats_over.clone().detach(),
                        self.queue_targets_over.clone().detach()
                    )
                else:
                    neighbor_tar = self.neighbor_targets(
                        outputs["proj_feats_novel"][:, :, ~mask_base, :],
                        self.queue_feats.clone().detach(),
                        self.queue_targets.clone().detach()
                    )
                    neighbor_tar_over = self.neighbor_targets(
                        outputs["proj_feats_novel_over"][:, :, ~mask_base, :],
                        self.queue_feats_over.clone().detach(),
                        self.queue_targets_over.clone().detach()
                    )

                targets[:, :, ~mask_base, nbc:] = self.sharpen(
                    self.hparams.queue_alpha * targets[:, :, ~mask_base, nbc:].type_as(targets) +
                    (1 - self.hparams.queue_alpha) * neighbor_tar.type_as(targets)
                ).type_as(targets)
                targets_over[:, :, ~mask_base, nbc:] = self.sharpen(
                    self.hparams.queue_alpha * targets_over[:, :, ~mask_base, nbc:].type_as(targets) +
                    (1 - self.hparams.queue_alpha) * neighbor_tar_over.type_as(targets)
                ).type_as(targets)

                if self.hparams.batch_head:
                    targets_batch_novel[:, :, :, nbc:] = targets[:, :, ~mask_base, nbc:]
                    targets_batch_novel_over[:, :, :, nbc:] = targets_over[:, :, ~mask_base, nbc:]
                    targets_batch[:, :, ~mask_base, :] = targets_batch_novel
                    targets_batch_over[:, :, ~mask_base, :] = targets_batch_novel_over

        # compute losses
        loss_cluster = self.swapped_prediction(logits, targets)
        loss_overcluster = self.swapped_prediction(logits_over, targets_over)
        if self.hparams.batch_head:
            loss_batch_cluster = self.swapped_prediction(logits_batch, targets_batch)
            loss_batch_overcluster = self.swapped_prediction(logits_batch_over, targets_batch_over)
            if self.hparams.batch_head_reg:
                loss_batch_base_reg = torch.norm(logits_batch_base[:, :, :, nbc:], dim=None)
                loss_batch_novel_reg = torch.norm(logits_batch_novel[:, :, :, :nbc], dim=None)
                loss_batch_novel_over_reg = torch.norm(logits_batch_novel_over[:, :, :, :nbc], dim=None)

        # update best head tracker, note that head with the smallest loss is not always the best
        self.loss_per_head += loss_cluster.clone().detach()
        if self.hparams.batch_head_multi_novel:
            self.loss_per_batch_head += loss_batch_cluster.clone().detach()

        # total loss and log
        loss = (loss_cluster + loss_overcluster) / 2
        results = {
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "loss_cluster": loss_cluster.mean(),
            "loss_overcluster": loss_overcluster.mean(),
        }
        if self.hparams.batch_head:
            loss += (loss_batch_cluster + loss_batch_overcluster) / 2
            results.update(
                {
                    "loss_batch_cluster": loss_batch_cluster.mean(),
                    "loss_batch_overcluster": loss_batch_overcluster.mean(),
                }
            )
            if self.hparams.batch_head_reg:
                loss += self.hparams.batch_head_reg * (
                            loss_batch_base_reg + loss_batch_novel_reg + loss_batch_novel_over_reg) / 3
                results.update(
                    {
                        "loss_batch_base_reg": loss_batch_base_reg.mean(),
                        "loss_batch_novel_reg": loss_batch_novel_reg.mean(),
                        "loss_batch_novel_over_reg": loss_batch_novel_over_reg.mean(),
                    }
                )
        results.update({"loss": loss.detach()})
        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def index_swap(self, i1, i2):
        index = torch.arange(self.hparams.num_heads)
        index[i1] = i2
        index[i2] = i1
        return index

    def validation_step(self, batch, batch_idx, dl_idx):
        images, labels = batch
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]
        nbc = self.hparams.num_base_classes

        # forward
        outputs = self.model(images)
        # logits_base:       [bs, base_class_num]
        # logits_novel:      [head_num, bs, novel_class_num]
        # logits_novel_over: [head_num, bs, 3*novel_class_num]
        # logits:            [head_num, bs, base_class_num + novel_class_num]
        # logits_over:       [head_num, bs, base_class_num+3*novel_class_num]

        if "novel" in tag:  # use clustering head
            preds = outputs["logits_novel"]
            if self.hparams.batch_head:
                if self.hparams.batch_head_multi_novel:
                    head_swapped = self.index_swap(torch.argmin(self.loss_per_head),
                                                   torch.argmin(self.loss_per_batch_head))
                    preds += self.hparams.alpha * outputs["logits_batch_novel"][head_swapped][:, :, nbc:]
                else:
                    preds += self.hparams.alpha * outputs["logits_batch_novel"].unsqueeze(0)[:, :, nbc:]
            preds_inc = torch.cat(  # incremental, task-agnostic actually
                [
                    outputs["logits_base"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_novel"],
                ],
                dim=-1,
            )
            if self.hparams.batch_head:
                preds_inc += self.hparams.alpha * outputs["logits_batch_base"].unsqueeze(0).expand(
                    self.hparams.num_heads, -1, -1)
                if self.hparams.batch_head_multi_novel:
                    head_swapped = self.index_swap(torch.argmin(self.loss_per_head),
                                                   torch.argmin(self.loss_per_batch_head))
                    preds_inc += self.hparams.alpha * outputs["logits_batch_novel"][head_swapped]
                else:
                    preds_inc += self.hparams.alpha * outputs["logits_batch_novel"].unsqueeze(0).expand(
                        self.hparams.num_heads, -1, -1)
        else:  # use supervised classifier
            preds = outputs["logits_base"]
            if self.hparams.big_head:
                preds += self.hparams.alpha * outputs["logits_batch_base"][:, :nbc]
            best_head = torch.argmin(self.loss_per_head)
            preds_inc = torch.cat(  # incremental, task-agnostic actually
                [outputs["logits_base"], outputs["logits_novel"][best_head]], dim=-1
            )
            if self.hparams.batch_head:
                preds_inc += self.hparams.alpha * outputs["logits_batch_base"]
                if self.hparams.batch_head_multi_novel:
                    best_batch_head = torch.argmin(self.loss_per_batch_head)
                    preds_inc += self.hparams.alpha * outputs["logits_batch_novel"][best_batch_head]
                else:
                    preds_inc += self.hparams.alpha * outputs["logits_batch_novel"]
        preds = preds.max(dim=-1)[1]
        preds_inc = preds_inc.max(dim=-1)[1]

        self.metrics[dl_idx].update(preds, labels)
        self.metrics_inc[dl_idx].update(preds_inc, labels)

    def validation_epoch_end(self, _):
        results = [m.compute() for m in self.metrics]
        results_inc = [m.compute() for m in self.metrics_inc]
        # log metrics
        for dl_idx, (result, result_inc) in enumerate(zip(results, results_inc)):
            prefix = self.trainer.datamodule.dataloader_mapping[dl_idx]
            prefix_inc = "incremental/" + prefix
            if "novel" in prefix:
                for (metric, values), (_, values_inc) in zip(result.items(), result_inc.items()):
                    name = "/".join([prefix, metric])
                    name_inc = "/".join([prefix_inc, metric])
                    avg = torch.stack(values).mean()
                    avg_inc = torch.stack(values_inc).mean()
                    best = values[torch.argmin(self.loss_per_head)]
                    best_inc = values_inc[torch.argmin(self.loss_per_head)]
                    self.log(name + "/avg", avg, sync_dist=True)
                    self.log(name + "/best", best, sync_dist=True)
                    self.log(name_inc + "/avg", avg_inc, sync_dist=True)
                    self.log(name_inc + "/best", best_inc, sync_dist=True)
            else:
                self.log(prefix + "/acc", result)
                self.log(prefix_inc + "/acc", result_inc)


def main(args):
    dm = get_datamodule(args, "discover")

    run_name = "-".join(["discover", args.arch, args.dataset, args.comment])
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    model = Discoverer(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_classes = args.num_base_classes + args.num_novel_classes

    main(args)
