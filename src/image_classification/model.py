"""Step 5 - The CNN."""
import torch.nn as nn
 
 
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )
 
 
class DefectCNN(nn.Module):
    """Input: 1 x 128 x 128 grayscale image  ->  Output: 2 scores (DEFECTIVE, OK)."""
 
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),     # 128 -> 64
            conv_block(16, 32),    # 64  -> 32
            conv_block(32, 64),    # 32  -> 16
            conv_block(64, 128),   # 16  -> 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),          # 128 x 4 x 4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )
 
    def forward(self, x):
        return self.classifier(self.features(x))
 
 
if __name__ == "__main__":
    import torch
    m = DefectCNN()
    print(m(torch.randn(2, 1, 128, 128)).shape)   # expect torch.Size([2, 2])
    print("Parameters:", sum(p.numel() for p in m.parameters()))
