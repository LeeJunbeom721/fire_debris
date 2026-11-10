import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 64, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(64),  # InstanceNorm1d -> BatchNorm1d로 수정
            nn.ReLU(),
            nn.MaxPool1d(4), 
            
            # Layer 2
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128), # InstanceNorm1d -> BatchNorm1d로 수정
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Layer 3
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256), # InstanceNorm1d -> BatchNorm1d로 수정
            nn.ReLU(),
            nn.MaxPool1d(4), 
            
            # Layer 4
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512), # InstanceNorm1d -> BatchNorm1d로 수정
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 1, Length)
        features = self.feature_extractor(x)
        return self.classifier(features)