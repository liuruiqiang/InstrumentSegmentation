import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
# from block import *rang
from transformer_block import position_embedding
from token_transformer import Token_transformer
import torch.nn.functional as F
from einops import rearrange, repeat
from RAUNet import AAM,DecoderBlockLinkNet
from trans import ViTS,ViT_FusionFrame
from Models.swin_transformer import BasicLayer
from Models.ViTViT import ViViT,Transformer,Transformer_fusion,\
    Transformer_fusionv2,Transformer_fusionv3,Transformer_Onlyfusion,\
    FusionTransformerWthDrloc,FusionTransformerWthTimeDrloc

class Conv2dReLU(nn.Module):
    """
    [Conv2d(in_channels, out_channels, kernel),
    BatchNorm2d(out_channels),
    ReLU,]
    """
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, bn=False):
        super(Conv2dReLU, self).__init__()
        modules = OrderedDict()
        modules['conv'] = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        if bn:
            modules['bn'] = nn.BatchNorm2d(out_channels)
        modules['relu'] = nn.ReLU(inplace=True)
        self.l = nn.Sequential(modules)

    def forward(self, x):
        x = self.l(x)
        return x


class UNetModule(nn.Module):
    """

    [Conv2dReLU(in_channels, out_channels, 3),
    Conv2dReLU(out_channels, out_channels, 3)]
    """
    def __init__(self, in_channels, out_channels, padding=1, bn=False):
        super(UNetModule, self).__init__()
        self.l = nn.Sequential(OrderedDict([
            ('conv1', Conv2dReLU(in_channels, out_channels, 3, padding=padding, bn=bn)),
            ('conv2', Conv2dReLU(out_channels, out_channels, 3, padding=padding, bn=bn))
            ]))

    def forward(self, x):
        x = self.l(x)
        return x

class Interpolate(nn.Module):
    """
    Wrapper function of interpolate/UpSample Module
    """
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.fn = lambda x: nn.functional.interpolate(x, scale_factor=scale_factor,
            mode=mode, align_corners=align_corners)

    def forward(self, x):
        return self.fn(x)


class UNet(nn.Module):
    """
    UNet implementation
    pretrained model: None
    Note: this implementation doesn't strictly follow UNet, the kernel sizes are halfed
    """
    def __init__(self, in_channels, num_classes, bn=False):
        super(UNet, self).__init__()

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample = Interpolate(scale_factor=2,
             mode='bilinear', align_corners=False)
        self.upsample4 = Interpolate(scale_factor=4,
             mode='bilinear', align_corners=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool4 = nn.MaxPool2d(4, stride=4)

        self.conv1 = UNetModule(in_channels, 32, bn=bn)
        self.conv2 = UNetModule(32, 64, bn=bn)
        self.conv3 = UNetModule(64, 128, bn=bn)
        self.conv4 = UNetModule(128, 256, bn=bn)
        self.center = UNetModule(256, 512, bn=bn)
        self.up4 = UNetModule(512 + 256, 256)
        self.up3 = UNetModule(256 + 128, 128)
        self.up2 = UNetModule(128 + 64, 64)
        self.up1 = UNetModule(64 + 32, 32)
        # final layer are logits
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, **kwargs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        center = self.center(self.pool4(conv4))

        up4 = self.up4(torch.cat([conv4, self.upsample4(center)], 1))
        up3 = self.up3(torch.cat([conv3, self.upsample(up4)], 1))
        up2 = self.up2(torch.cat([conv2, self.upsample(up3)], 1))
        up1 = self.up1(torch.cat([conv1, self.upsample(up2)], 1))

        output = self.final(up1)
        return output


class DecoderModule(nn.Module):
    """
    DecoderModule for UNet11, UNet16

    Upsample version:
    [Interpolate(scale_factor, 'bilinear'),
    Con2dReLU(in_channels, mid_channels),
    Conv2dReLU(mid_channels, out_channels),
    ]

    DeConv version:
    [Con2dReLU(in_channels, mid_channels),
    ConvTranspose2d(mid_channels, out_channels, kernel=4, stride=2, pad=1),
    ReLU
    ]
    """
    def __init__(self, in_channels, mid_channels, out_channels, upsample=True):
        super(DecoderModule, self).__init__()
        if upsample:
            modules = OrderedDict([
                ('interpolate', Interpolate(scale_factor=2, mode='bilinear',
                    align_corners=False)),
                ('conv1', Conv2dReLU(in_channels, mid_channels)),
                ('conv2', Conv2dReLU(mid_channels, out_channels))
                ])
        else:
            modules = OrderedDict([
                ('conv', Conv2dReLU(in_channels, mid_channels)),
                ('deconv', nn.ConvTranspose2d(mid_channels,
                    out_channels, kernel_size=4, stride=2, padding=1)),
                ('relu', nn.ReLU(inplace=True))
                ])
        self.l = nn.Sequential(modules)

    def forward(self, x):
        return self.l(x)


class UNet11(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self,  num_classes,in_channels=3, pretrained=True, bn=False, upsample=False):
        super(UNet11, self).__init__()
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)
        self.num_classes = num_classes

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, **kwargs):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        center = self.center(self.pool(conv5)) # 256

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        output = self.final(dec1)
        if self.num_classes > 1:
            # print('!!')
            # print('.....')
            x_out = F.log_softmax(output, dim=1)
        else:
            x_out = output
        return x_out

class Conv2dReLU1x1(nn.Module):
    """
    [Conv2d(in_channels, out_channels, kernel),
    BatchNorm2d(out_channels),
    ReLU,]
    """
    def __init__(self, in_channels, out_channels, kernel=1, padding=0, bn=False):
        super(Conv2dReLU1x1, self).__init__()
        modules = OrderedDict()
        modules['conv'] = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        if bn:
            modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['in'] = nn.InstanceNorm2d(out_channels)
        modules['relu'] = nn.ReLU(inplace=True)
        self.l = nn.Sequential(modules)

    def forward(self, x):
        x = self.l(x)
        return x

class MFFM(nn.Module):
    """
    Multi frame fusion module
    """
    def __init__(self,in_channel,mode='add'):
        super(MFFM, self).__init__()
        self.mode = mode
        if mode == 'add':
            self.fusion = Conv2dReLU1x1(in_channel,in_channel*3)
        else:
            self.fusion = Conv2dReLU1x1(in_channel*3,in_channel*3)


    def forward(self,x,t=3):
        """
        Args:
            x: b,c,h,w(t,c,h,w)
        Returns:
        """
        b,c,h,w = x.shape
        # print(b, t)
        x = x.view(int(b/t),t,c,h,w)

        if(self.mode == 'add'):
            x = torch.sum(x,dim=1)
        else:
            items = torch.unbind(x,dim=1)
            # print(len(items),items[0].shape)
            x = torch.cat(items,dim=1)
            # print(x.shape)
        x = self.fusion(x)
        x = x.view(b,c,h,w)
        return x

# data = torch.rand(3,512,14,14).cuda()
# model = MFFM(512,mode='cat').cuda()
# print(model(data).shape)
class TeRAUNet_ViT(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=320,w=256
                 ,numheads=12,depth=4):
        super().__init__()
        assert num_channels == 3
        self.w = h
        self.h = w
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # self.fusion4 = MFFM(512,mode='cat')

        self.trans = ViTS(img_dim=h//2**5,
             in_channels=filters[3],  # encoder channels
             patch_dim=2,
             dim=filters[3]*4,  # vit out channels for decoding
             blocks=depth,  # self.vit_blocks,
             heads=numheads,  # self.vit_heads,
             dim_linear_block=128,
             classification=False).cuda()

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        self.gau3 = AAM(filters[2], filters[2]) #RAUNet
        self.gau2 = AAM(filters[1], filters[1])
        self.gau1 = AAM(filters[0], filters[0])


        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        s1 = x.size() #input data size:1,b,c,h,w
        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b,c,h,w = e4.shape
        e4 = e4.view(b//3,3,c,h,w)
        e4 = self.trans(e4)
        e4 = e4.view(b,c,h,w)

        # e4 = self.fusion4(e4)
        d4 = self.decoder4(e4)
        b4 = self.gau3(d4, e3)
        d3 = self.decoder3(b4)
        b3 = self.gau2(d3, e2)
        d2 = self.decoder2(b3)
        b2 = self.gau1(d2, e1)
        d1 = self.decoder1(b2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out

class TeRAUNet_ViTFusion(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=320,w=256
                 ,numheads=12,depth = 4):
        super().__init__()
        assert num_channels == 3
        self.w = h
        self.h = w
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # self.fusion4 = MFFM(512,mode='cat')

        self.trans = ViT_FusionFrame(img_dim=h//2**5,
             in_channels=filters[3],  # encoder channels
             patch_dim=2,
             dim=filters[3]*4,  # vit out channels for decoding
             blocks=depth,  # self.vit_blocks,
             heads=numheads,  # self.vit_heads,
             dim_linear_block=128,
             classification=False).cuda()

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        self.gau3 = AAM(filters[2], filters[2]) #RAUNet
        self.gau2 = AAM(filters[1], filters[1])
        self.gau1 = AAM(filters[0], filters[0])


        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        s1 = x.size() #input data size:1,b,c,h,w
        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b,c,h,w = e4.shape
        e4 = e4.view(b//3,3,c,h,w)
        e4 = self.trans(e4)
        e4 = e4.view(b,c,h,w)

        # e4 = self.fusion4(e4)
        d4 = self.decoder4(e4)
        b4 = self.gau3(d4, e3)
        d3 = self.decoder3(b4)
        b3 = self.gau2(d3, e2)
        d2 = self.decoder2(b3)
        b2 = self.gau1(d2, e1)
        d1 = self.decoder1(b2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out


class MultiFrameFusionUNet11_v2(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False, upsample=False,
                 h=224,w=224,):
        super(MultiFrameFusionUNet11_v2, self).__init__()
        self.h = h
        self.w = w
        self.num_classes = num_classes
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.fusion1 = MFFM(128)
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.fusion2 = MFFM(256)
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.fusion3 = MFFM(512)
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]
        self.fusion4 = MFFM(512)

        # self.TimeTrans = Token_transformer((h // 2 ** 5) * (w // 2 ** 5), (h // 2 ** 5) * (w // 2 ** 5), num_heads=numheads)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, **kwargs):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        conv2_fused = self.fusion1(conv2)
        # print(conv2.shape)
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv3_fused = self.fusion2(conv3)
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv4_fused = self.fusion3(conv4)
        conv5 = self.conv5(self.pool(conv4)) # 512
        conv5_fused = self.fusion4(conv5)
        center = self.center(self.pool(conv5)) # 256
        # print(conv3.shape,conv4.shape,conv5.shape,center.shape)

        dec5 = self.dec5(torch.cat([center, conv5_fused], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4_fused], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3_fused], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2_fused], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        output = self.final(dec1)
        if self.num_classes > 1:
            x_out = F.log_softmax(output, dim=1)
        else:
            x_out = output
        return x_out

class MultiFrameFusionUNet11(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False, upsample=False,
                 h=224,w=224,numheads = 4):
        super(MultiFrameFusionUNet11, self).__init__()
        self.h = h
        self.w = w
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        # position embedding
        self.pose_emb = position_embedding(3 * (h // 2 ** 5) * (w // 2 ** 5), 512)
        #
        # # transformer
        self.SpatialTrans = Token_transformer(512, 512, num_heads=numheads)
        # self.TimeTrans = Token_transformer((h // 2 ** 5) * (w // 2 ** 5), (h // 2 ** 5) * (w // 2 ** 5), num_heads=numheads)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, **kwargs):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        # print(conv2.shape)
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        trans_input = self.pool(conv5)

        # s4 = trans_input.size()
        #MultiFrameFusion
        b,c,h, w = trans_input.size()
        # print('trans_shape:', trans_input.shape)
        trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, 'b t c h w -> b (t h w) c')
        posed_trans_input = self.pose_emb(trans_input)
        spatial_out = self.SpatialTrans(posed_trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> b t c h w', t=3, h=h, w=w)
        trans_out = trans_out.squeeze(0)

        center = self.center(trans_out) # 256
        # print(conv3.shape,conv4.shape,conv5.shape,center.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        # output = self.final(dec1)
        output = self.final(dec1)
        if self.num_classes > 1:
            x_out = F.log_softmax(output, dim=1)
        else:
            x_out = output
        return x_out

class MultiFrameFusionUNet11_swin(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False, upsample=False,
                 h=224,w=224,numheads = 4):
        super(MultiFrameFusionUNet11_swin, self).__init__()
        self.h = h
        self.w = w
        self.num_classes = num_classes
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        # position embedding
        # self.pose_emb = position_embedding(3 * (h // 2 ** 5) * (w // 2 ** 5), 512)
        #
        # # transformer
        self.swin_trans = BasicLayer(dim=512,input_resolution=(h // 2 ** 5,w // 2 ** 5),depth=3,num_heads=numheads,
                                     window_size=2,indim=512)
        # self.SpatialTrans = Token_transformer(512, 512, num_heads=numheads)
        # self.TimeTrans = Token_transformer((h // 2 ** 5) * (w // 2 ** 5), (h // 2 ** 5) * (w // 2 ** 5), num_heads=numheads)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, **kwargs):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        # print(conv2.shape)
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        trans_input = self.pool(conv5)

        # s4 = trans_input.size()
        #MultiFrameFusion
        b,c,h, w = trans_input.size()
        # print('trans_shape:', trans_input.shape)
        # trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, 'b c h w -> b (h w) c')
        spatial_out = self.swin_trans(trans_input)
        trans_out = rearrange(spatial_out, 'b (h w) c -> b c h w', h=h, w=w)
        # trans_out = trans_out.squeeze(0)

        center = self.center(trans_out) # 256
        # print(conv3.shape,conv4.shape,conv5.shape,center.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        # output = self.final(dec1)
        output = self.final(dec1)
        if self.num_classes > 1:
            x_out = F.log_softmax(output, dim=1)
        else:
            x_out = output
        return x_out

class MultiFrameFusionUNet11_ViTViT(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, num_classes, pretrained=True, bn=False, upsample=False,
                 h=224,w=224,numheads = 4,time_length=3):
        super(MultiFrameFusionUNet11_ViTViT, self).__init__()
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        # position embedding
        # self.pose_emb = position_embedding(3 * (h // 2 ** 5) * (w // 2 ** 5), 512)
        #
        # # transformer
        self.ViTViT = Transformer(dim=512,heads=numheads,dim_head=512,mlp_dim=512*4,dropout=0.1,
                    num_patches_space=(h // 2 ** 5) * (w // 2 ** 5),num_patches_time=time_length)
        # self.SpatialTrans = Token_transformer(512, 512, num_heads=numheads)
        # self.TimeTrans = Token_transformer((h // 2 ** 5) * (w // 2 ** 5), (h // 2 ** 5) * (w // 2 ** 5), num_heads=numheads)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, return_map=False):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        # print(conv2.shape)
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        trans_input = self.pool(conv5)

        # s4 = trans_input.size()
        #MultiFrameFusion
        b,c,h, w = trans_input.size()
        trans_inputforshow = trans_input
        # print('trans_shape:', trans_input.shape)
        # trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, '(b t) c h w -> b (t h w) c',t=self.time_length)
        spatial_out = self.ViTViT(trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> (b t) c h w', t=self.time_length,h=h, w=w)
        # trans_out = trans_out.squeeze(0)

        center = self.center(trans_out) # 256
        # print(conv3.shape,conv4.shape,conv5.shape,center.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        # output = self.final(dec1)
        output = self.final(dec1)
        if self.num_classes > 1:
            x_out = F.log_softmax(output, dim=1)
        else:
            x_out = output
        if return_map:
            return x_out,trans_inputforshow,trans_out
        return x_out


class MultiFrameFusionUNet11_ViTViTv2(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, num_classes, pretrained=True, bn=False, upsample=False,
                 h=224,w=224,numheads = 4,time_length=3):
        super(MultiFrameFusionUNet11_ViTViTv2, self).__init__()
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        # position embedding
        # self.pose_emb = position_embedding(3 * (h // 2 ** 5) * (w // 2 ** 5), 512)
        #
        # # transformer
        self.ViTViT = Transformer_fusion(dim=512,heads=numheads,dim_head=512,mlp_dim=512*4,dropout=0.1,
                    num_patches_space=(h // 2 ** 5) * (w // 2 ** 5),num_patches_time=self.time_length)
        # self.SpatialTrans = Token_transformer(512, 512, num_heads=numheads)
        # self.TimeTrans = Token_transformer((h // 2 ** 5) * (w // 2 ** 5), (h // 2 ** 5) * (w // 2 ** 5), num_heads=numheads)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, return_map=False):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        # print(conv2.shape)
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        trans_input = self.pool(conv5)

        # s4 = trans_input.size()
        #MultiFrameFusion
        b,c,h, w = trans_input.size()
        trans_inputforshow = trans_input
        # print('trans_shape:', trans_input.shape)
        # trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, '(b t) c h w -> b (t h w) c',t=self.time_length)
        spatial_out = self.ViTViT(trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> (b t) c h w', t=self.time_length,h=h, w=w)
        # trans_out = trans_out.squeeze(0)

        center = self.center(trans_out) # 256
        # print(conv3.shape,conv4.shape,conv5.shape,center.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        # output = self.final(dec1)
        output = self.final(dec1)
        if self.num_classes > 1:
            x_out = F.log_softmax(output, dim=1)
        else:
            x_out = output
        if return_map:
            return x_out,trans_inputforshow,trans_out
        return x_out

class MultiFrameFusionUNet11_ViTViTv3(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, num_classes, pretrained=True, bn=False, upsample=False,
                 h=224,w=224,numheads = 4,time_length=3):
        super(MultiFrameFusionUNet11_ViTViTv3, self).__init__()
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        # position embedding
        # self.pose_emb = position_embedding(3 * (h // 2 ** 5) * (w // 2 ** 5), 512)
        #
        # # transformer
        self.ViTViT = Transformer_fusionv2(dim=512,heads=numheads,dim_head=512,mlp_dim=512*4,dropout=0.1,
                    num_patches_space=(h // 2 ** 5) * (w // 2 ** 5),num_patches_time=time_length)
        # self.SpatialTrans = Token_transformer(512, 512, num_heads=numheads)
        # self.TimeTrans = Token_transformer((h // 2 ** 5) * (w // 2 ** 5), (h // 2 ** 5) * (w // 2 ** 5), num_heads=numheads)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, return_map=False,t=3):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        # print(conv2.shape)
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        trans_input = self.pool(conv5)

        # s4 = trans_input.size()
        #MultiFrameFusion
        b,c,h, w = trans_input.size()
        trans_inputforshow = trans_input
        # print('trans_shape:', trans_input.shape)
        # trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, '(b t) c h w -> b (t h w) c',t=self.time_length)
        spatial_out = self.ViTViT(trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> (b t) c h w', t=self.time_length,h=h, w=w)
        # trans_out = trans_out.squeeze(0)

        center = self.center(trans_out) # 256
        # print(conv3.shape,conv4.shape,conv5.shape,center.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        # output = self.final(dec1)
        output = self.final(dec1)
        if self.num_classes > 1:
            x_out = F.log_softmax(output, dim=1)
        else:
            x_out = output
        if return_map:
            return x_out,trans_inputforshow,trans_out
        return x_out

class MultiFrameFusionUNet11_ViTViTv4(nn.Module):
    """
    space attention and fusion attention
    pretrained-model: ImageNet
    """
    def __init__(self, num_classes, pretrained=True, bn=False, upsample=False,
                 h=224,w=224,numheads = 4,time_length=3):
        super(MultiFrameFusionUNet11_ViTViTv4, self).__init__()
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        # position embedding
        # self.pose_emb = position_embedding(3 * (h // 2 ** 5) * (w // 2 ** 5), 512)
        #
        # # transformer
        self.ViTViT = Transformer_fusionv3(dim=512,heads=numheads,dim_head=512,mlp_dim=512*4,dropout=0.1,
                    num_patches_space=(h // 2 ** 5) * (w // 2 ** 5),num_patches_time=time_length)
        # self.SpatialTrans = Token_transformer(512, 512, num_heads=numheads)
        # self.TimeTrans = Token_transformer((h // 2 ** 5) * (w // 2 ** 5), (h // 2 ** 5) * (w // 2 ** 5), num_heads=numheads)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, return_map=False,t=3):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        # print(conv2.shape)
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        trans_input = self.pool(conv5)

        # s4 = trans_input.size()
        #MultiFrameFusion
        b,c,h, w = trans_input.size()
        trans_inputforshow = trans_input
        # print('trans_shape:', trans_input.shape)
        # trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, '(b t) c h w -> b (t h w) c',t=self.time_length)
        spatial_out = self.ViTViT(trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> (b t) c h w', t=self.time_length,h=h, w=w)
        # trans_out = trans_out.squeeze(0)

        center = self.center(trans_out) # 256
        # print(conv3.shape,conv4.shape,conv5.shape,center.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        # output = self.final(dec1)
        output = self.final(dec1)
        if self.num_classes > 1:
            x_out = F.log_softmax(output, dim=1)
        else:
            x_out = output
        if return_map:
            return x_out,trans_inputforshow,trans_out
        return x_out

class MultiFrameFusionUNet11_OnlyFusion(nn.Module):
    """
    only fusion attention
    pretrained-model: ImageNet
    """
    def __init__(self, num_classes, pretrained=True, bn=False, upsample=False,
                 h=224,w=224,numheads = 4,time_length=3):
        super(MultiFrameFusionUNet11_OnlyFusion, self).__init__()
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        # position embedding
        # self.pose_emb = position_embedding(3 * (h // 2 ** 5) * (w // 2 ** 5), 512)
        #
        # # transformer
        self.ViTViT = Transformer_Onlyfusion(dim=512,heads=numheads,dim_head=512,mlp_dim=512*4,dropout=0.1,
                    num_patches_space=(h // 2 ** 5) * (w // 2 ** 5),num_patches_time=time_length)
        # self.SpatialTrans = Token_transformer(512, 512, num_heads=numheads)
        # self.TimeTrans = Token_transformer((h // 2 ** 5) * (w // 2 ** 5), (h // 2 ** 5) * (w // 2 ** 5), num_heads=numheads)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, return_map=False,t=3):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        # print(conv2.shape)
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        trans_input = self.pool(conv5)

        # s4 = trans_input.size()
        #MultiFrameFusion
        b,c,h, w = trans_input.size()
        trans_inputforshow = trans_input
        # print('trans_shape:', trans_input.shape)
        # trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, '(b t) c h w -> b (t h w) c',t=self.time_length)
        spatial_out = self.ViTViT(trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> (b t) c h w', t=self.time_length,h=h, w=w)
        # trans_out = trans_out.squeeze(0)

        center = self.center(trans_out) # 256
        # print(conv3.shape,conv4.shape,conv5.shape,center.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        # output = self.final(dec1)
        output = self.final(dec1)
        if self.num_classes > 1:
            x_out = F.log_softmax(output, dim=1)
        else:
            x_out = output
        if return_map:
            return x_out,trans_inputforshow,trans_out
        return x_out


class MultiFrameFusionUNet11_OnlyFusionWthDrloc(nn.Module):
    """
    only fusion attention
    pretrained-model: ImageNet
    """
    def __init__(self, num_classes, pretrained=True, bn=False, upsample=False,
                 h=224,w=224,numheads = 4,time_length=3,useDrl=False,drloc_mode="l1",sample_size=32,
        use_abs=True,):
        super(MultiFrameFusionUNet11_OnlyFusionWthDrloc, self).__init__()
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        self.useDrl = useDrl
        # position embedding
        # self.pose_emb = position_embedding(3 * (h // 2 ** 5) * (w // 2 ** 5), 512)
        #
        # # transformer
        self.transformer = FusionTransformerWthDrloc(dim=512,heads=numheads,dim_head=512,mlp_dim=512*4,dropout=0.1,
                    num_patches_space=(h // 2 ** 5) * (w // 2 ** 5),num_patches_time=time_length,
                                                     useDrl=useDrl, drloc_mode=drloc_mode,
                                                     sample_size=sample_size,
                                                     use_abs=use_abs,
                                                     )
        # self.SpatialTrans = Token_transformer(512, 512, num_heads=numheads)
        # self.TimeTrans = Token_transformer((h // 2 ** 5) * (w // 2 ** 5), (h // 2 ** 5) * (w // 2 ** 5), num_heads=numheads)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, return_map=False,t=3):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        # print(conv2.shape)
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        trans_input = self.pool(conv5)

        # s4 = trans_input.size()
        #MultiFrameFusion
        b,c,h, w = trans_input.size()
        trans_inputforshow = trans_input
        # print('trans_shape:', trans_input.shape)
        # trans_input = trans_input.view(1,b,c,h,w)
        if self.useDrl:
            trans_out,drlout = self.transformer(trans_input)
        else:
            trans_out = self.transformer(trans_input)
        # trans_out = rearrange(spatial_out, 'b (t h w) c -> (b t) c h w', t=self.time_length,h=h, w=w)
        # trans_out = trans_out.squeeze(0)

        center = self.center(trans_out) # 256
        # print(conv3.shape,conv4.shape,conv5.shape,center.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        # output = self.final(dec1)
        output = self.final(dec1)
        if self.num_classes > 1:
            x_out = F.log_softmax(output, dim=1)
        else:
            x_out = output
        if return_map:
            return x_out,trans_inputforshow,trans_out
        if self.useDrl:
            return x_out,drlout
        return x_out

class RAU_MultiFrameFusion_Net(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,
                 h=224,w=224,numheads = 4,time_length=3):
        super().__init__()
        assert num_channels == 3
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        self.gau3 = AAM(filters[2], filters[2]) #RAUNet
        self.gau2 = AAM(filters[1], filters[1])
        self.gau1 = AAM(filters[0], filters[0])

        self.ViTViT = Transformer_Onlyfusion(dim=512, heads=numheads, dim_head=512, mlp_dim=512 * 4, dropout=0.1,
                                             num_patches_space=(h // 2 ** 5) * (w // 2 ** 5), num_patches_time=time_length)


        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        s1 = x.size() #input data size:1,b,c,h,w
        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        trans_input = e4

        # s4 = trans_input.size()
        # MultiFrameFusion
        b, c, h, w = trans_input.size()
        trans_inputforshow = trans_input
        # print('trans_shape:', trans_input.shape)
        # trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, '(b t) c h w -> b (t h w) c', t=self.time_length)
        spatial_out = self.ViTViT(trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> (b t) c h w', t=self.time_length, h=h, w=w)

        d4 = self.decoder4(trans_out)
        b4 = self.gau3(d4, e3)
        d3 = self.decoder3(b4)
        b3 = self.gau2(d3, e2)
        d2 = self.decoder2(b3)
        b2 = self.gau1(d2, e1)
        d1 = self.decoder1(b2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out

class RAU_MultiFrameFusion_Net_WtoAttn(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,
                 h=224,w=224,numheads = 4,time_length=3):
        super().__init__()
        assert num_channels == 3
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        # self.gau3 = AAM(filters[2], filters[2]) #RAUNet
        # self.gau2 = AAM(filters[1], filters[1])
        # self.gau1 = AAM(filters[0], filters[0])

        self.ViTViT = Transformer_Onlyfusion(dim=512, heads=numheads, dim_head=512, mlp_dim=512 * 4, dropout=0.1,
                                             num_patches_space=(h // 2 ** 5) * (w // 2 ** 5), num_patches_time=time_length)


        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        s1 = x.size() #input data size:1,b,c,h,w
        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        print(e2.shape)
        e3 = self.encoder3(e2)
        print(e3.shape)
        e4 = self.encoder4(e3)
        print(e4.shape)
        trans_input = e4

        # s4 = trans_input.size()
        # MultiFrameFusion
        b, c, h, w = trans_input.size()
        trans_inputforshow = trans_input
        # print('trans_shape:', trans_input.shape)
        # trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, '(b t) c h w -> b (t h w) c', t=self.time_length)
        spatial_out = self.ViTViT(trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> (b t) c h w', t=self.time_length, h=h, w=w)

        d4 = self.decoder4(trans_out)
        b4 = torch.add(d4,e3)
        d3 = self.decoder3(b4)
        b3 = torch.add(d3, e2)
        d2 = self.decoder2(b3)
        b2 = torch.add(d2, e1)
        d1 = self.decoder1(b2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out

class Res_MultiFrameFusion_Net_Wdrloc(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,
                 h=224,w=224,numheads = 4,time_length=3,useDrl=False,drloc_mode="l1",sample_size=32,
        use_abs=True,):
        super().__init__()
        assert num_channels == 3
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        self.useDrl = useDrl
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        # self.gau3 = AAM(filters[2], filters[2]) #RAUNet
        # self.gau2 = AAM(filters[1], filters[1])
        # self.gau1 = AAM(filters[0], filters[0])

        self.transformer = FusionTransformerWthDrloc(dim=512, heads=numheads, dim_head=512, mlp_dim=512 * 4, dropout=0.1,
                                                     num_patches_space=(h // 2 ** 5) * (w // 2 ** 5), num_patches_time=time_length,
                                                     useDrl=useDrl, drloc_mode=drloc_mode,
                                                     sample_size=sample_size,
                                                     use_abs=use_abs,
                                                     )

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        s1 = x.size() #input data size:1,b,c,h,w
        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        trans_input = e4

        # s4 = trans_input.size()
        # MultiFrameFusion
        if self.useDrl:
            trans_out, drlout = self.transformer(trans_input)
        else:
            trans_out = self.transformer(trans_input)
        d4 = self.decoder4(trans_out)
        b4 = torch.add(d4,e3)
        d3 = self.decoder3(b4)
        b3 = torch.add(d3, e2)
        d2 = self.decoder2(b3)
        b2 = torch.add(d2, e1)
        d1 = self.decoder1(b2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        if self.useDrl:
            return x_out,drlout
        return x_out

class Res_MultiFrameFusion_Net_WTimedrloc(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,
                 h=224,w=224,numheads = 4,time_length=3,useDrl=False,drloc_mode="l1",sample_size=32,
        use_abs=True,):
        super().__init__()
        assert num_channels == 3
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        self.useDrl = useDrl
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        # self.gau3 = AAM(filters[2], filters[2]) #RAUNet
        # self.gau2 = AAM(filters[1], filters[1])
        # self.gau1 = AAM(filters[0], filters[0])

        self.transformer = FusionTransformerWthTimeDrloc(dim=512, heads=numheads, dim_head=512, mlp_dim=512 * 4, dropout=0.1,
                                                     num_patches_space=(h // 2 ** 5) * (w // 2 ** 5), num_patches_time=time_length,
                                                     useDrl=useDrl, drloc_mode=drloc_mode,
                                                     sample_size=sample_size,
                                                     use_abs=use_abs,
                                                     )

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        s1 = x.size() #input data size:1,b,c,h,w
        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        trans_input = e4

        # s4 = trans_input.size()
        # MultiFrameFusion
        if self.useDrl:
            trans_out, drlout = self.transformer(trans_input)
        else:
            trans_out = self.transformer(trans_input)
        d4 = self.decoder4(trans_out)
        b4 = torch.add(d4,e3)
        d3 = self.decoder3(b4)
        b3 = torch.add(d3, e2)
        d2 = self.decoder2(b3)
        b2 = torch.add(d2, e1)
        d1 = self.decoder1(b2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        if self.useDrl:
            return x_out,drlout
        return x_out

class RAU_ViViT_Net(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,
                 h=224,w=224,numheads = 4,time_length=3):
        super().__init__()
        assert num_channels == 3
        self.h = h
        self.w = w
        self.num_classes = num_classes
        self.time_length = time_length
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        # filters = [256, 512, 1024, 2048]
        # resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        self.gau3 = AAM(filters[2], filters[2]) #RAUNet
        self.gau2 = AAM(filters[1], filters[1])
        self.gau1 = AAM(filters[0], filters[0])

        # # transformer
        self.ViTViT = Transformer(dim=512, heads=numheads, dim_head=512, mlp_dim=512 * 4, dropout=0.1,
                                  num_patches_space=(h // 2 ** 5) * (w // 2 ** 5), num_patches_time=time_length)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        s1 = x.size() #input data size:1,b,c,h,w
        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        trans_input = e4

        # s4 = trans_input.size()
        # MultiFrameFusion
        b, c, h, w = trans_input.size()
        trans_inputforshow = trans_input
        # print('trans_shape:', trans_input.shape)
        # trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, '(b t) c h w -> b (t h w) c', t=self.time_length)
        spatial_out = self.ViTViT(trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> (b t) c h w', t=self.time_length, h=h, w=w)

        d4 = self.decoder4(trans_out)
        b4 = self.gau3(d4, e3)
        d3 = self.decoder3(b4)
        b3 = self.gau2(d3, e2)
        d2 = self.decoder2(b3)
        b2 = self.gau1(d2, e1)
        d1 = self.decoder1(b2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out

class TSTransUNet11(nn.Module):
    """
    UNet11: use VGG11 as encoder and corresponding DecoderModule as decoder
    pretrained-model: ImageNet
    """
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False, upsample=False,
                 h=224,w=224,numheads = 4):
        super(TSTransUNet11, self).__init__()
        self.h = h
        self.w = w
        if bn:
            self.vgg11 = models.vgg11_bn(pretrained=pretrained).features
            pool_idxs = [3, 7, 14, 21, 28]
        else:
            self.vgg11 = models.vgg11(pretrained=pretrained).features
            pool_idxs = [2, 5, 10, 15, 20]

        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = self.vgg11[0:pool_idxs[0]]
        self.conv2 = self.vgg11[pool_idxs[0]+1:pool_idxs[1]]
        self.conv3 = self.vgg11[pool_idxs[1]+1:pool_idxs[2]]
        self.conv4 = self.vgg11[pool_idxs[2]+1:pool_idxs[3]]
        self.conv5 = self.vgg11[pool_idxs[3]+1:pool_idxs[4]]

        # position embedding
        self.pose_emb = position_embedding(3 * (h // 2 ** 5) * (w // 2 ** 5), 512)
        #
        # # transformer
        self.SpatialTrans = Token_transformer(3*512, 3*512, num_heads=numheads)
        self.TimeTrans = Token_transformer((h // 2 ** 5) * (w // 2 ** 5), (h // 2 ** 5) * (w // 2 ** 5), num_heads=numheads)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(512 + 256, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(512 + 256, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(256 + 128, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(128 + 64, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(64 + 32, 32)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, **kwargs):
        s1 = x.size()  # input data size:1,b,c,h,w
        x = x.view([s1[0] * s1[1], s1[2], s1[3], s1[4]])  # 1,b,c,h,w -> b,c,h,w
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.pool(conv1)) # 128
        print(conv2.shape)
        conv3 = self.conv3(self.pool(conv2)) # 256
        conv4 = self.conv4(self.pool(conv3)) # 512
        conv5 = self.conv5(self.pool(conv4)) # 512
        trans_input = self.pool(conv5)

        # s4 = trans_input.size()
        b,c,h, w = trans_input.size()

        trans_input = trans_input.view(1,b,c,h,w)
        trans_input = rearrange(trans_input, 'b t c h w -> b (t h w) c')
        posed_trans_input = self.pose_emb(trans_input)
        spatial_input = rearrange(posed_trans_input, 'b (t h w) c -> b (h w) (t c)', h=h, w=w)
        spatial_out = self.SpatialTrans(spatial_input)
        trans_out = rearrange(spatial_out, 'b (h w) (t c) -> b t c h w', t=3, h=h, w=w)
        trans_out = trans_out.squeeze(0)

        center = self.center(trans_out) # 256
        # print(conv3.shape,conv4.shape,conv5.shape,center.shape)

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 256
        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 128
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 32
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 32

        output = self.final(dec1)
        return output
#
data = torch.rand(1,3,3,320,256).cuda()
# data = torch.rand(2,5,320,256)
# data2 = data[:,::2,:,:]
# print('data2',data2.shape)
model = RAU_MultiFrameFusion_Net_WtoAttn(3,3,h=320,w=256).cuda()
# model = MultiFrameFusionUNet11_OnlyFusionWthDrloc(3,3,h=320,w=256,numheads=4,useDrl=True).cuda()
# # model = UNet11(3,3)
# model.cuda()
out = model(data)
print(out.shape)
# # print(model(data).shape)
# data = torch.rand(1,3,3,640,512).cuda()
# model = MultiFrameFusionUNet11_v2(3,3,h=640,w=512)
# model = UNet11(3,3)
# model.cuda()
# out = model(data)
# print(model(data).shape)
# print(out)
# print(out)
# spatial_out = torch.rand(1,14*14,3*512)
# trans_out = rearrange(spatial_out, 'b (h w) (t c) -> b t c h w', t=3, h=14, w=14)
# last_out = rearrange(trans_out,'b t c h w -> b (h w) (t c)')
# print(last_out.shape,trans_out.shape)
# print(last_out == spatial_out)