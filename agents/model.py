import torch
import torch.nn as nn
import torch.nn.functional as F

class WallzNet(nn.Module):
    def __init__(self, num_channels=8):
        super().__init__()
        # Convolutional extraction core targeting spatial 9x9 configuration layers
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Output Action Policy Allocation (Size 209 space map)
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 9 * 9, 209)
        
        # Scalar Position Value Estimation State Head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x, action_mask=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Extract Actor Logit Spaces
        p = F.relu(self.policy_conv(x))
        p = p.view(-1, 32 * 9 * 9)
        logits = self.policy_fc(p)
        
        if action_mask is not None:
            # Enforce hard logic masks onto the unnormalized policy distribution bounds
            logits = logits.masked_fill(~action_mask, -1e9)
            
        # Extract State Values evaluations
        v = F.relu(self.value_conv(x))
        v = v.view(-1, 9 * 9)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return logits, value
