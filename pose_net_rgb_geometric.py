"""RGB pose estimation with geometric translation using pinhole camera model."""

import torch
import torch.nn as nn
import torchvision.models as models


class PoseNetRGBGeometric(nn.Module):
    """
    Pose estimation model using RGB input with geometric translation.
    Learns rotation and Z-depth; computes X, Y using pinhole camera model.
    """
    
    def __init__(self, pretrained=True):
        super(PoseNetRGBGeometric, self).__init__()
        
        # RGB backbone for rotation
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Rotation head
        self.rot_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4)
        )
        
        # Lightweight CNN for Z-depth prediction
        self.z_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Z-depth predictor
        self.z_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize Z bias to typical depth
        self.z_predictor[-1].bias.data.fill_(0.5)

    def forward(self, rgb, bbox_center=None, camera_matrix=None):
        """Forward pass with optional geometric translation."""
        # Rotation from ResNet50
        rgb_features = self.rgb_backbone(rgb).view(rgb.size(0), -1)
        rotation = self.rot_head(rgb_features)
        rotation = rotation / (torch.norm(rotation, dim=1, keepdim=True) + 1e-8)
        
        # Z-depth from lightweight CNN
        z_features = self.z_backbone(rgb)
        z_pred = self.z_predictor(z_features)
        
        # Compute X, Y using pinhole model if camera info provided
        if bbox_center is not None and camera_matrix is not None:
            translation = self._compute_pinhole_translation(z_pred, bbox_center, camera_matrix)
        else:
            translation = torch.cat([
                torch.zeros_like(z_pred),
                torch.zeros_like(z_pred),
                z_pred
            ], dim=1)
        
        return rotation, translation

    def _compute_pinhole_translation(self, z_pred, bbox_center, camera_matrix):
        """Compute X, Y from Z using pinhole camera model."""
        if camera_matrix.dim() == 2:
            camera_matrix = camera_matrix.unsqueeze(0).expand(z_pred.size(0), -1, -1)
        
        fx = camera_matrix[:, 0, 0].unsqueeze(1)
        fy = camera_matrix[:, 1, 1].unsqueeze(1)
        cx = camera_matrix[:, 0, 2].unsqueeze(1)
        cy = camera_matrix[:, 1, 2].unsqueeze(1)
        
        u = bbox_center[:, 0].unsqueeze(1)
        v = bbox_center[:, 1].unsqueeze(1)
        
        x_pred = (u - cx) * z_pred / fx
        y_pred = (v - cy) * z_pred / fy
        
        return torch.cat([x_pred, y_pred, z_pred], dim=1)
