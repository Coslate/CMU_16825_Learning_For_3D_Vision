import torch

class UncertaintyMLP(torch.nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
        self.apply(self.init_weights)  # [ADDED] initialize weights

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)        

# Tiny UNet for per-pixel confidence prediction
class TinyUNet(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.ReLU()
        )
        self.pool1 = torch.nn.MaxPool2d(2)

        self.encoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU()
        )
        self.pool2 = torch.nn.MaxPool2d(2)

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU()
        )

        self.up2 = torch.nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU()
        )

        self.up1 = torch.nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.ReLU()
        )

        self.final = torch.nn.Conv2d(16, out_channels, 1)
        self._init_weights()  # [ADDED] apply initialization

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)        

    def forward(self, x):
        e1 = self.encoder1(x) #(B, 16, H, W)
        p1 = self.pool1(e1) #(B, 16, H/2, W/2)
        e2 = self.encoder2(p1) #(B, 32, H/2, W/2)
        p2 = self.pool2(e2) #(B, 32, H/4, W/4)
        b = self.bottleneck(p2) #(B, 64, H/4, W/4)
        u2 = self.up2(b) #(B, 32, H/2, W/2)
        d2 = self.decoder2(torch.cat([u2, e2], dim=1)) #(B, 32, H/2, W/2)
        u1 = self.up1(d2) #(B, 16, H, W)
        d1 = self.decoder1(torch.cat([u1, e1], dim=1)) #(B, 16, H, W)
        return torch.sigmoid(self.final(d1)) #(B, 1, H, W)

# Replace ConfidenceMLP with TinyUNet in training loop accordingly
