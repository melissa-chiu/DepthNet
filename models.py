import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print(x.shape, skip_input.shape)
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Generator_no_rgb(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator_no_rgb, self).__init__()

        # encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # depth-mask share weights decoder
        self.dm_up1 = UNetUp(512, 512, dropout=0.5)
        self.dm_up2 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up3 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up4 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up5 = UNetUp(1024, 256)
        self.dm_up6 = UNetUp(512, 128)
        self.dm_up7 = UNetUp(256, 64)

        self.depth_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )
        self.mask_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 7, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # depth-mask decoder
        dm_u1 = self.dm_up1(d8, d7)
        dm_u2 = self.dm_up2(dm_u1, d6)
        dm_u3 = self.dm_up3(dm_u2, d5)
        dm_u4 = self.dm_up4(dm_u3, d4)
        dm_u5 = self.dm_up5(dm_u4, d3)
        dm_u6 = self.dm_up6(dm_u5, d2)
        dm_u7 = self.dm_up7(dm_u6, d1)
        depth = self.depth_final(dm_u7)
        mask = self.mask_final(dm_u7)
        return depth, mask, d8


class Generator_no_m(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator_no_m, self).__init__()

        # encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # depth decoder
        self.dm_up1 = UNetUp(512, 512, dropout=0.5)
        self.dm_up2 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up3 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up4 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up5 = UNetUp(1024, 256)
        self.dm_up6 = UNetUp(512, 128)
        self.dm_up7 = UNetUp(256, 64)

        self.depth_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

        # rgb decoder
        self.rgb_up1 = UNetUp(512, 512, dropout=0.5)
        self.rgb_up2 = UNetUp(1024, 512, dropout=0.5)
        self.rgb_up3 = UNetUp(1024, 512, dropout=0.5)
        self.rgb_up4 = UNetUp(1024, 512, dropout=0.5)
        self.rgb_up5 = UNetUp(1024, 256)
        self.rgb_up6 = UNetUp(512, 128)
        self.rgb_up7 = UNetUp(256, 64)
        self.rgb_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # depth decoder
        dm_u1 = self.dm_up1(d8, d7)
        dm_u2 = self.dm_up2(dm_u1, d6)
        dm_u3 = self.dm_up3(dm_u2, d5)
        dm_u4 = self.dm_up4(dm_u3, d4)
        dm_u5 = self.dm_up5(dm_u4, d3)
        dm_u6 = self.dm_up6(dm_u5, d2)
        dm_u7 = self.dm_up7(dm_u6, d1)
        depth = self.depth_final(dm_u7)

        # rgb decoder
        rgb_u1 = self.rgb_up1(d8, d7)
        rgb_u2 = self.rgb_up2(rgb_u1, d6)
        rgb_u3 = self.rgb_up3(rgb_u2, d5)
        rgb_u4 = self.rgb_up4(rgb_u3, d4)
        rgb_u5 = self.rgb_up5(rgb_u4, d3)
        rgb_u6 = self.rgb_up6(rgb_u5, d2)
        rgb_u7 = self.rgb_up7(rgb_u6, d1)
        rgb = self.rgb_final(rgb_u7)
        return rgb, depth, d8


class Generator_no_rgbm(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator_no_rgbm, self).__init__()

        # encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # depth decoder
        self.dm_up1 = UNetUp(512, 512, dropout=0.5)
        self.dm_up2 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up3 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up4 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up5 = UNetUp(1024, 256)
        self.dm_up6 = UNetUp(512, 128)
        self.dm_up7 = UNetUp(256, 64)

        self.depth_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # depth decoder
        dm_u1 = self.dm_up1(d8, d7)
        dm_u2 = self.dm_up2(dm_u1, d6)
        dm_u3 = self.dm_up3(dm_u2, d5)
        dm_u4 = self.dm_up4(dm_u3, d4)
        dm_u5 = self.dm_up5(dm_u4, d3)
        dm_u6 = self.dm_up6(dm_u5, d2)
        dm_u7 = self.dm_up7(dm_u6, d1)
        depth = self.depth_final(dm_u7)
        return depth, d8


class DM_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DM_Generator, self).__init__()

        # encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # depth-mask share weights decoder
        self.dm_up1 = UNetUp(512, 512, dropout=0.5)
        self.dm_up2 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up3 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up4 = UNetUp(1024, 512, dropout=0.5)
        self.dm_up5 = UNetUp(1024, 256)
        self.dm_up6 = UNetUp(512, 128)
        self.dm_up7 = UNetUp(256, 64)

        self.depth_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )
        self.mask_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 7, 4, padding=1),
            nn.Tanh(),
        )

        # rgb decoder
        self.rgb_up1 = UNetUp(512, 512, dropout=0.5)
        self.rgb_up2 = UNetUp(1024, 512, dropout=0.5)
        self.rgb_up3 = UNetUp(1024, 512, dropout=0.5)
        self.rgb_up4 = UNetUp(1024, 512, dropout=0.5)
        self.rgb_up5 = UNetUp(1024, 256)
        self.rgb_up6 = UNetUp(512, 128)
        self.rgb_up7 = UNetUp(256, 64)
        self.rgb_final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # depth-mask decoder
        dm_u1 = self.dm_up1(d8, d7)
        dm_u2 = self.dm_up2(dm_u1, d6)
        dm_u3 = self.dm_up3(dm_u2, d5)
        dm_u4 = self.dm_up4(dm_u3, d4)
        dm_u5 = self.dm_up5(dm_u4, d3)
        dm_u6 = self.dm_up6(dm_u5, d2)
        dm_u7 = self.dm_up7(dm_u6, d1)
        depth = self.depth_final(dm_u7)
        mask = self.mask_final(dm_u7)

        # rgb decoder
        rgb_u1 = self.rgb_up1(d8, d7)
        rgb_u2 = self.rgb_up2(rgb_u1, d6)
        rgb_u3 = self.rgb_up3(rgb_u2, d5)
        rgb_u4 = self.rgb_up4(rgb_u3, d4)
        rgb_u5 = self.rgb_up5(rgb_u4, d3)
        rgb_u6 = self.rgb_up6(rgb_u5, d2)
        rgb_u7 = self.rgb_up7(rgb_u6, d1)
        rgb = self.rgb_final(rgb_u7)
        return rgb, depth, mask, d8


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A):
        return self.model(img_A)

if __name__=='__main__':
    img = torch.Tensor(1, 4, 256, 256)
    gt = torch.Tensor(1, 3, 256, 256)
    d = Discriminator(in_channels=7)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img, gt = img.cuda(), gt.cuda()
    d = d.cuda()

    # calculate the inference time:
    d.eval()
    with torch.no_grad():
        for i in range(100):
            x = d(img, gt)
            print(x.shape)
            print(x[0, 0])
            break
            