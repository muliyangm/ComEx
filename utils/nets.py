import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=-1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1
    ):
        super().__init__()
        self.num_heads = num_heads

        # projectors
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(output_dim, num_prototypes) for _ in range(num_heads)]
        )
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=-1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadResNet(nn.Module):
    """
    head_base               : base-class expert
    head_novel(_over)       : novel-class expert
    head_batch_base         : base-batch expert
    head_batch_novel(_over) : novel-batch expert
    """
    def __init__(
        self,
        arch,
        low_res,
        num_base,
        num_novel,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3,
        num_heads=5,
        num_hidden_layers=1,
        batch_head=False,
        batch_head_multi_novel=False
    ):
        super().__init__()

        self.batch_head_multi_novel = batch_head_multi_novel

        # backbone
        self.encoder = models.__dict__[arch]()
        self.feat_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        # modify the encoder for lower resolution
        if low_res:
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()
            self._reinit_all_layers()

        self.head_base = Prototypes(self.feat_dim, num_base)

        if batch_head:
            self.head_batch_base = Prototypes(self.feat_dim, num_base + num_novel)
            self.head_batch_novel = Prototypes(self.feat_dim, num_base + num_novel)
            self.head_batch_novel_over = Prototypes(self.feat_dim, num_base + num_novel*overcluster_factor)

        if num_heads is not None:
            self.head_novel = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_novel,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
            self.head_novel_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_novel * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
            if batch_head_multi_novel and batch_head:
                self.head_batch_novel = MultiHead(
                    input_dim=self.feat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=proj_dim,
                    num_prototypes=num_base+num_novel,
                    num_heads=num_heads,
                    num_hidden_layers=num_hidden_layers,
                )
                self.head_batch_novel_over = MultiHead(
                    input_dim=self.feat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=proj_dim,
                    num_prototypes=num_base + num_novel*overcluster_factor,
                    num_heads=num_heads,
                    num_hidden_layers=num_hidden_layers,
                )

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_base.normalize_prototypes()
        if hasattr(self, "head_batch_base"):
            self.head_batch_base.normalize_prototypes()
            self.head_batch_novel.normalize_prototypes()
            self.head_batch_novel_over.normalize_prototypes()
        if hasattr(self, "head_novel"):
            self.head_novel.normalize_prototypes()
            self.head_novel_over.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"logits_base": self.head_base(F.normalize(feats, dim=-1))}
        out.update({"feats_base": F.normalize(feats, dim=-1)})
        if hasattr(self, "head_batch_base"):
            out.update({"logits_batch_base": self.head_batch_base(F.normalize(feats, dim=-1))})
            if self.batch_head_multi_novel:
                logits_batch_novel, proj_feats_batch_novel = self.head_batch_novel(feats)
                logits_batch_novel_over, proj_feats_batch_novel_over = self.head_batch_novel_over(feats)
                out.update(
                    {
                        "logits_batch_novel": logits_batch_novel,
                        "proj_feats_batch_novel": proj_feats_batch_novel,
                        "logits_batch_novel_over": logits_batch_novel_over,
                        "proj_feats_batch_novel_over": proj_feats_batch_novel_over,
                    }
                )
            else:  # linear classifier if not multi head
                out.update(
                    {
                        "logits_batch_novel": self.head_batch_novel(F.normalize(feats, dim=-1)),
                        "logits_batch_novel_over": self.head_batch_novel_over(F.normalize(feats, dim=-1)),
                    }
                )
        if hasattr(self, "head_novel"):
            logits_novel, proj_feats_novel = self.head_novel(feats)
            logits_novel_over, proj_feats_novel_over = self.head_novel_over(feats)
            out.update(
                {
                    "logits_novel": logits_novel,
                    "proj_feats_novel": proj_feats_novel,
                    "logits_novel_over": logits_novel_over,
                    "proj_feats_novel_over": proj_feats_novel_over,
                }
            )
        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder(views)
            out = self.forward_heads(feats)
            out["feats"] = feats
            return out
