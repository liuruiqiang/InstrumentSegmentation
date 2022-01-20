from torch import nn
from torchvision import models
import torch.nn.functional as F
from transformer_block import position_embedding
from token_transformer import Token_transformer
from einops import rearrange
from Models.network_bcloks import MFFM
from Models.network_bcloks import TransformerBlock,FocalTransformerBlock


class AAM(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(AAM, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.Softmax(dim=1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input_high, input_low):
        mid_high=self.global_pooling(input_high)
        weight_high=self.conv1(mid_high)

        mid_low = self.global_pooling(input_low)
        weight_low = self.conv2(mid_low)

        weight=self.conv3(weight_low+weight_high)
        low = self.conv4(input_low)
        return input_high+low.mul(weight)

class RAUNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.w = 512
        self.h = 640
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
        s4=e4.size()
        # print(s4)
        e4=e4.view([s1[0],s1[1],s4[1],s4[2],s4[3]]) #b,c,h,w->1,b,c,h,w

        # transformer code
        e4=e4.view([s1[0]*s1[1],s4[1],s4[2],s4[3]]) # 1,b,c,h,w -> b,c,h,w
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

class TransRAUNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=320,w=256,numheads=4):
        super().__init__()
        assert num_channels == 3
        self.w = w
        self.h = h
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

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        self.gau3 = AAM(filters[2], filters[2]) #RAUNet
        self.gau2 = AAM(filters[1], filters[1])
        self.gau1 = AAM(filters[0], filters[0])

        # position embedding
        self.pose_emb = position_embedding((h // 2 ** 5) * (w // 2 ** 5), 512)
        # print('h,w',3 * (h // 2 ** 5) * (w // 2 ** 5))
        # # transformer
        self.SpatialTrans = Token_transformer(filters[3], filters[3], num_heads=numheads)


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
        s4=e4.size()
        # print(s4)
        e4=e4.permute(0,2,3,1) #b,c,h,w->1,b,c,h,w
        e4 = e4.view(b,h*w,c)
        e4 = self.pose_emb(e4)
        e4 = self.SpatialTrans(e4)
        e4 = e4.view(b,h,w,c).permute(0,3,1,2)

        # transformer code
        # e4=e4.view([s1[0]*s1[1],s4[1],s4[2],s4[3]]) # 1,b,c,h,w -> b,c,h,w
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

class MultiFrameFusionRAUNet_v2(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=320,w=256):
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
        self.fusion1 = MFFM(64)
        self.encoder2 = resnet.layer2
        self.fusion2 = MFFM(128)
        self.encoder3 = resnet.layer3
        self.fusion3 = MFFM(256)
        self.encoder4 = resnet.layer4
        self.fusion4 = MFFM(512)

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
        e1_fused = self.fusion1(e1)
        e2 = self.encoder2(e1)
        e2_fused = self.fusion2(e2)
        e3 = self.encoder3(e2)
        e3_fused = self.fusion3(e3)
        e4 = self.encoder4(e3)
        e4 = self.fusion4(e4)
        s4=e4.size()
        # print(s4)
        e4=e4.view([s1[0],s1[1],s4[1],s4[2],s4[3]]) #b,c,h,w->1,b,c,h,w

        # transformer code
        e4=e4.view([s1[0]*s1[1],s4[1],s4[2],s4[3]]) # 1,b,c,h,w -> b,c,h,w
        d4 = self.decoder4(e4)
        b4 = self.gau3(d4, e3_fused)
        d3 = self.decoder3(b4)
        b3 = self.gau2(d3, e2_fused)
        d2 = self.decoder2(b3)
        b2 = self.gau1(d2, e1_fused)
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

class MultiFrameFusionRAUNet_v3(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=320,w=256):
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
        self.fusion4 = MFFM(512,mode='cat')

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
        e4 = self.fusion4(e4)
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


class TeRAUNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=224,w=224,numheads = 4):
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

        #position embedding
        self.pose_emb = position_embedding(num_channels*(h//2**5)*(w//2**5),filters[3])

        #transformer
        self.SpatialTrans = Token_transformer(filters[3],filters[3],num_heads=numheads)
        self.TimeTrans = Token_transformer(filters[3], filters[3], num_heads=numheads)

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
        s4=e4.size()
        h,w = s4[2],s4[3]
        # print(s4)
        e4=e4.view([s1[0],s1[1],s4[1],s4[2],s4[3]]) #b,c,h,w->1,b,c,h,w
        e4 = rearrange(e4,'b t c h w -> b (t h w) c')
        posed_e4 = self.pose_emb(e4)
        time_input = rearrange(posed_e4,'b (t h w) c -> (b h w) t c',h=h,w=w)
        # print(time_input.shape)
        time_out = self.TimeTrans(time_input)
        # print(time_out.shape)
        spatial_input = rearrange(time_out,'(b h w) t c -> (b t) (h w) c',h=h,w=w)
        spatial_out = self.SpatialTrans(spatial_input)
        # print(spatial_out.shape)
        e4 = rearrange(spatial_out,'(b t) (h w) c -> b t c h w',t=s1[1],h=h,w=w)

        # print(e4.shape)
        # transformer code
        e4=e4.view([s1[0]*s1[1],s4[1],s4[2],s4[3]]) # 1,b,c,h,w -> b,c,h,w
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

class MultiFrameFusionRAUNet11(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=224,w=224,numheads = 4):
        super().__init__()
        assert num_channels == 3
        self.w = h
        self.h = w
        # print('h,w',h,w)
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

        # position embedding
        self.pose_emb = position_embedding(3 * (h // 2 ** 5) * (w // 2 ** 5), 512)
        # print('h,w',3 * (h // 2 ** 5) * (w // 2 ** 5))
        #
        # # transformer
        self.SpatialTrans = Token_transformer(filters[3],filters[3], num_heads=numheads)
        # self.SpatialTrans = Token_transformer(filters[3],filters[3],num_heads=numheads)
        # self.TimeTrans = Token_transformer(filters[3], filters[3], num_heads=numheads)

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
    def forward(self, x,return_map=False):
        s1 = x.size() #input data size:1,b,c,h,wRP

        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> 1*b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        s4=e4.size()
        h,w = s4[2],s4[3]
        # print(s4)
        b, c, h, w = e4.size()
        t = b//s1[0]
        # print('trans_shape:', trans_input.shape)
        trans_input = e4.view(s1[0], t, c, h, w)
        trans_input = rearrange(trans_input, 'b t c h w -> b (t h w) c')
        posed_trans_input = self.pose_emb(trans_input)
        spatial_out = self.SpatialTrans(posed_trans_input)
        trans_out = rearrange(spatial_out, 'b (t h w) c -> b t c h w', t=3, h=h, w=w)
        trans_out = trans_out.view(s1[0]*s1[1],c,h,w)

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
        if return_map:
            return x_out,e4,trans_out
        return x_out

class MultiFrameFusionRAUNet11_2trans(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=224,w=224,numheads = 4):
        super().__init__()
        assert num_channels == 3
        self.w = h
        self.h = w
        # print('h,w',h,w)
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


        self.boottle_transformer = TransformerBlock(3 * (h // 2 ** 5) * (w // 2 ** 5),filters[3],num_heads=numheads)
        self.trans2 = TransformerBlock(3 * (h // 2 ** 4) * (w // 2 ** 4),filters[2],num_heads=numheads)
        # self.SpatialTrans = Token_transformer(filters[3],filters[3],num_heads=numheads)
        # self.TimeTrans = Token_transformer(filters[3], filters[3], num_heads=numheads)

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
        s1 = x.size() #input data size:1,b,c,h,wRP

        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> 1*b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        s4=e4.size()

        e4_trans_out = self.boottle_transformer(e4)

        d4 = self.decoder4(e4_trans_out)

        e3_trans_out = self.trans2(e3)
        b4 = self.gau3(d4,e3_trans_out)
        # print(d4.shape,e3.shape,b4.shape)
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

class FocalTransRAUNet11_2Trans(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=224,w=224,numheads = 4):
        super().__init__()
        assert num_channels == 3
        self.w = h
        self.h = w
        # print('h,w',h,w)
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


        self.boottle_transformer = FocalTransformerBlock((h // 2 ** 5) * (w // 2 ** 5),filters[3],num_heads=numheads,num_levels=2,
                                                         input_resolution=(h // 2 ** 5,w // 2 ** 5))
        self.trans2 = FocalTransformerBlock((h // 2 ** 4) * (w // 2 ** 4),filters[2],num_heads=numheads,num_levels=3,
                                            input_resolution=(h // 2 ** 4,w // 2 ** 4))
        # self.SpatialTrans = Token_transformer(filters[3],filters[3],num_heads=numheads)
        # self.TimeTrans = Token_transformer(filters[3], filters[3], num_heads=numheads)

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
        s1 = x.size() #input data size:1,b,c,h,wRP

        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> 1*b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        s4=e4.size()

        e4_trans_out = self.boottle_transformer(e4)

        d4 = self.decoder4(e4_trans_out)

        e3_trans_out = self.trans2(e3)
        b4 = self.gau3(d4,e3_trans_out)
        # print(d4.shape,e3.shape,b4.shape)
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

class FocalTransRAUNet11(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=224,w=224,numheads = 4):
        super().__init__()
        assert num_channels == 3
        self.w = h
        self.h = w
        # print('h,w',h,w)
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


        self.boottle_transformer = FocalTransformerBlock((h // 2 ** 5) * (w // 2 ** 5),filters[3],num_heads=numheads,num_levels=2,
                                                         input_resolution=(h // 2 ** 5,w // 2 ** 5))
        # self.trans2 = FocalTransformerBlock((h // 2 ** 4) * (w // 2 ** 4),filters[2],num_heads=numheads,num_levels=3,
        #                                     input_resolution=(h // 2 ** 4,w // 2 ** 4))
        # self.SpatialTrans = Token_transformer(filters[3],filters[3],num_heads=numheads)
        # self.TimeTrans = Token_transformer(filters[3], filters[3], num_heads=numheads)

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
        s1 = x.size() #input data size:1,b,c,h,wRP

        x=x.view([s1[0]*s1[1],s1[2],s1[3],s1[4]]) # 1,b,c,h,w -> 1*b,c,h,w
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        s4=e4.size()

        e4_trans_out = self.boottle_transformer(e4)
        d4 = self.decoder4(e4_trans_out)
        b4 = self.gau3(d4, e3)
        # print(d4.shape,e3.shape,b4.shape)
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

class TSTransRAUNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,h=224,w=224,numheads = 4):
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

        #position embedding
        self.pose_emb = position_embedding(num_channels*(h//2**5)*(w//2**5),filters[3])

        #transformer
        self.SpatialTrans = Token_transformer(filters[3],filters[3],num_heads=numheads)
        self.TimeTrans = Token_transformer(filters[3], filters[3], num_heads=numheads)

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
        s4=e4.size()
        h,w = s4[2],s4[3]
        # print(s4)
        e4=e4.view([s1[0],s1[1],s4[1],s4[2],s4[3]]) #b,c,h,w->1,b,c,h,w
        e4 = rearrange(e4,'b t c h w -> b (t h w) c')
        posed_e4 = self.pose_emb(e4)
        time_input = rearrange(posed_e4,'b (t h w) c -> (b h w) t c',h=h,w=w)
        # print(time_input.shape)
        time_out = self.TimeTrans(time_input)
        # print(time_out.shape)
        spatial_input = rearrange(time_out,'(b h w) t c -> (b t) (h w) c',h=h,w=w)
        spatial_out = self.SpatialTrans(spatial_input)
        # print(spatial_out.shape)
        e4 = rearrange(spatial_out,'(b t) (h w) c -> b t c h w',t=s1[1],h=h,w=w)

        # print(e4.shape)
        # transformer code
        e4=e4.view([s1[0]*s1[1],s4[1],s4[2],s4[3]]) # 1,b,c,h,w -> b,c,h,w
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
        
class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x

import torch
data = torch.rand(2,3,3,320,256).cuda()
# model = TransRAUNet(num_classes=3,h=320,w=256)
# model.cuda()
# out = model(data)
# print(out.shape)
