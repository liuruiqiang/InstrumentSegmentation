import torch
import torch.nn as nn
import torch.nn.functional as F
from trans import ViTT,ViTS,SignleConv
from einops import rearrange
from timesformer_pytorch import TimeSformer
from convlstm import ConvLSTM
class SupConUnet(nn.Module):
    def __init__(self, num_classes, in_channels=3, initial_filter_size=64,
                 kernel_size=3, do_instancenorm=True, mode="cls"):
        super(SupConUnet, self).__init__()

        self.encoder = UNet(num_classes, in_channels, initial_filter_size, kernel_size, do_instancenorm)
        if mode == 'mlp':
            self.head = nn.Sequential(nn.Conv2d(initial_filter_size, 256, kernel_size=1),
                                      nn.Conv2d(256, num_classes, kernel_size=1))
        elif mode == "cls":
            self.head = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)
        else:
            raise NotImplemented("This mode is not supported yet")

    def forward(self, x):

        y = self.encoder(x)

        output = self.head(y)
        output = F.log_softmax(output, dim=1)
        # output = F.normalize(self.head(y), dim=1)

        return output


class SupConUnetInfer(nn.Module):
    def __init__(self, num_classes, in_channels=1, initial_filter_size=64,
                 kernel_size=3, do_instancenorm=True):
        super(SupConUnetInfer, self).__init__()

        self.encoder = UNet(num_classes, in_channels, initial_filter_size, kernel_size, do_instancenorm)
        self.head = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)

    def forward(self, x):

        y = self.encoder(x)
        print(y)
        output = self.head(y)

        # output = F.normalize(self.head(y), dim=1)

        return y, output


class UNet(nn.Module):

    def __init__(self, num_classes, in_channels=1, initial_filter_size=64, kernel_size=3, do_instancenorm=True):
        super().__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size*2, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.contr_3_1 = self.contract(initial_filter_size*2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size*2**2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        # self.pool3 = nn.MaxPool2d(2, stride=2)

        self.contr_4_1 = self.contract(initial_filter_size*2**2, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size*2**3, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size*2**3, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size*2**4, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size*2**4, initial_filter_size*2**3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size*2**4, initial_filter_size*2**3)
        self.expand_4_2 = self.expand(initial_filter_size*2**3, initial_filter_size*2**3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size*2**3, initial_filter_size*2**2, kernel_size=2, stride=2)

        self.expand_3_1 = self.expand(initial_filter_size*2**3, initial_filter_size*2**2)
        self.expand_3_2 = self.expand(initial_filter_size*2**2, initial_filter_size*2**2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size*2**2, initial_filter_size*2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size*2**2, initial_filter_size*2)
        self.expand_2_2 = self.expand(initial_filter_size*2, initial_filter_size*2)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size*2, initial_filter_size, 2, stride=2)

        self.expand_1_1 = self.expand(initial_filter_size*2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        # Output layer for segmentation
        # self.final = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)  # kernel size for final layer = 1, see paper

        self.softmax = torch.nn.Softmax2d()

        # Output layer for "autoencoder-mode"
        self.output_reconstruction_map = nn.Conv2d(initial_filter_size, out_channels=1, kernel_size=1)
        # self.vits = ViTS(img_dim=28,
        #                in_channels=512,  # encoder channels
        #                patch_dim=2,
        #                dim=2048,  # vit out channels for decoding
        #                blocks=2,#self.vit_blocks,
        #                heads=12,#self.vit_heads,
        #                dim_linear_block=128,
        #                classification=False)
        # self.vitt = ViTT(img_dim=14,
        #                  in_channels=512,  # encoder channels
        #                  patch_dim=1,
        #                  dim=512,  # vit out channels for decoding
        #                  blocks=6,
        #                  heads=12,
        #                  dim_linear_block=128,
        #                  classification=False)
        self.convlstm = ConvLSTM(
                                 input_dim=512,
                                 hidden_dim=[512,512],
                                 kernel_size=(3,3),
                                 num_layers=2,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
            )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x, enable_concat=True):
        s1 = x.size()
        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]])
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)
        x = pool.view([1, 3, 512, 14, 14])
        x,_=self.convlstm(x)
        # x = self.vitt(x)
        pool = x[0].view([3, 512, 14, 14])
        center = self.center(pool)


        # x=self.vits(x)
        # x=self.vitt(x)

        # center=x
        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat = torch.cat([center, crop*concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop*concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop*concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)

        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop*concat_weight], 1)

        expand = self.expand_1_2(self.expand_1_1(concat))

        return expand


class DownsampleUnet(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=64, kernel_size=3, do_instancenorm=True):
        super().__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size*2, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.contr_3_1 = self.contract(initial_filter_size*2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size*2**2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        # self.pool3 = nn.MaxPool2d(2, stride=2)

        self.contr_4_1 = self.contract(initial_filter_size*2**2, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size*2**3, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size*2**3, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size*2**4, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size*2**4, initial_filter_size*2**3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.softmax = torch.nn.Softmax2d()

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x, enable_concat=True):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        center = self.center(pool)

        return center


class GlobalConUnet(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=64):
        super().__init__()
        self.encoder = DownsampleUnet(in_channels, initial_filter_size)

    def forward(self, x):
        y = self.encoder(x)

        return y


class MLP(nn.Module):
    def __init__(self, input_channels=512, num_class=128):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Linear(input_channels, input_channels)
        self.f2 = nn.Linear(input_channels, num_class)

    def forward(self, x):
        x = self.gap(x)
        y = self.f1(x.squeeze())
        y = self.f2(y)

        return y


class UpsampleUnet2(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=64, kernel_size=3, do_instancenorm=True):
        super().__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size*2, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.contr_3_1 = self.contract(initial_filter_size*2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size*2**2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        # self.pool3 = nn.MaxPool2d(2, stride=2)

        self.contr_4_1 = self.contract(initial_filter_size*2**2, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size*2**3, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size*2**3, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size*2**4, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size*2**4, initial_filter_size*2**3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_2 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2,
                                           stride=2)

        self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size * 2 ** 2, initial_filter_size * 2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2)
        self.expand_2_2 = self.expand(initial_filter_size * 2, initial_filter_size * 2)
        self.softmax = torch.nn.Softmax2d()

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x, enable_concat=True):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        center = self.center(pool)

        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat = torch.cat([center, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat))

        return expand


class LocalConUnet2(nn.Module):
    def __init__(self, num_classes, in_channels=1, initial_filter_size=64):
        super().__init__()
        self.encoder = UpsampleUnet2(in_channels, initial_filter_size)
        self.head = MLP(input_channels=initial_filter_size*2, num_class=num_classes)

    def forward(self, x):
        y = self.encoder(x)

        return y


class UpsampleUnet3(nn.Module):
    def __init__(self, in_channels=1, initial_filter_size=64, kernel_size=3, do_instancenorm=True):
        super().__init__()

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size*2, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.contr_3_1 = self.contract(initial_filter_size*2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size*2**2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        # self.pool3 = nn.MaxPool2d(2, stride=2)

        self.contr_4_1 = self.contract(initial_filter_size*2**2, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size*2**3, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size*2**3, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size*2**4, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size*2**4, initial_filter_size*2**3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
        self.expand_4_2 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2,
                                           stride=2)

        self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
        self.expand_3_2 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)

        self.softmax = torch.nn.Softmax2d()

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x, enable_concat=True):
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        center = self.center(pool)

        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat = torch.cat([center, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat))

        return expand


class LocalConUnet3(nn.Module):
    def __init__(self, num_classes, in_channels=1, initial_filter_size=64):
        super().__init__()
        self.encoder = UpsampleUnet3(in_channels, initial_filter_size)
        self.head = MLP(input_channels=initial_filter_size*2*2, num_class=num_classes)

    def forward(self, x):
        y = self.encoder(x)

        return y


