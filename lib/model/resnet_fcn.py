import torch
import torch.nn as nn

from torch.nn import functional as F


class FCNHead(nn.Sequential):

    def __init__(
            self,
            in_channels, channels,
            dropout=0.1
    ):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


class ResNetFCN(nn.Module):

    def __init__(
            self,
            base_model_name='fcn_resnet50',
            output_size=128,
            n_out_channels=2,
            pretrained=True,
            use_pretrain_head=True,
            use_aux=False):
        super().__init__()
        # load torch hub resnet fcn
        self.base_model_name = base_model_name
        self.output_size = output_size
        self.n_out_channels = n_out_channels
        self.pretrained = pretrained
        self.use_pretrain_head = use_pretrain_head
        self.use_aux = use_aux

        self.base_model = self._get_base_model()
        self.backbone = self.base_model.backbone
        model_fcn_heads = self._get_fcn_heads()
        self.fcn_head = model_fcn_heads['classifier']
        self.fcn_aux = model_fcn_heads['aux_classifier']

    def _get_base_model(self):
        base_model = torch.hub.load(
            'pytorch/vision:v0.10.0',  # just hard code
            self.base_model_name,
            pretrained=self.pretrained
        )
        return base_model

    def _get_fcn_heads(self):
        if self.use_pretrain_head:
            self.base_model.classifier[4] = nn.Conv2d(
                512, 2,
                kernel_size=1,
                stride=1
            )
            self.base_model.aux_classifier[4] = nn.Conv2d(
                256, 2,
                kernel_size=1,
                stride=1
            )

        else:
            self.base_model.classifier = FCNHead(
                2048, self.n_out_channels,
                dropout=0.1
            )
            self.base_model.aux_classifier = FCNHead(
                1024, self.n_out_channels,
                dropout=0.1
            )

        output = {
            'classifier': self.base_model.classifier,
            'aux_classifier': self.base_model.aux_classifier if self.use_aux else None
        }

        return output

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        out = self.fcn_head(x['out'])
        aux = self.fcn_aux(x['aux']) if self.use_aux else None

        if self.output_size != 32:
            out = F.interpolate(out, size=self.output_size, mode='bilinear', align_corners=False)
            if aux is not None:
                aux = F.interpolate(aux, size=self.output_size, mode='bilinear', align_corners=False)

        output = {
            'out': out,
            'aux': aux
        }
        return output


if __name__ == '__main__':
    # always required CUDA please
    dummy_input = torch.randn((32, 3, 256, 256)).cuda()  # 4 x RGB input
    model = ResNetFCN(
        base_model_name='fcn_resnet50',
        output_size=128,
        n_out_channels=2,
        pretrained=True,
        use_pretrain_head=False,
        use_aux=True
    ).cuda()

    output = model(dummy_input)
    print(output['out'].size())
    if isinstance(output['aux'], torch.Tensor):
        print(output['aux'].size())
