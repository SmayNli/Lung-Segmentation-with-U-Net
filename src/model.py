from torch import nn
import torch

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.convblock1=nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )

        self.convblock2=nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )

        self.convblock3=nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )

        self.convblock4=nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU()
        )

        self.convblock5=nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 2, 2)
        )

        self.upconv1=nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2)
        )

        self.upconv2=nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2)
        )

        self.upconv3=nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2)
        )

        self.segmentation_convblock=nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def copy_tensor(self, contracting_tensor, expansive_tensor):

        return torch.cat([contracting_tensor, expansive_tensor], 1)

    def forward(self, x):
        x=self.convblock1(x)
        copy1=x
        x=self.convblock2(x)
        copy2=x
        x=self.convblock3(x)
        copy3=x
        x=self.convblock4(x)
        copy4=x
        x=self.convblock5(x)
        x=self.copy_tensor(copy4, x)
        x=self.upconv1(x)
        x=self.copy_tensor(copy3, x)
        x=self.upconv2(x)
        x=self.copy_tensor(copy2, x)
        x=self.upconv3(x)
        x=self.copy_tensor(copy1, x)
        x=self.segmentation_convblock(x)

        return x

if __name__=="__main__":

    sample_tensor=torch.rand(1, 1, 256, 256)
    model=UNET()

    sample_result=model(sample_tensor)

    print(sample_result.shape)


