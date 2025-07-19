import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    alexnet, AlexNet_Weights,
    vgg16, VGG16_Weights
)

class FeatureExtractor:
    def __init__(self, architecture='resnet', augment=False):
        """
        architecture: 'alexnet', 'vgg', 'resnet'
        augment: bool - If True, apply data augmentation.
        """
        self.architecture = architecture
        self.augment = augment
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(architecture)
        self.features = None
        self._register_hooks()
        self.transform = self._get_transforms(augment)

    def _load_model(self, architecture: str) -> nn.Module:
        if architecture == 'alexnet':
            model = alexnet(weights=AlexNet_Weights.DEFAULT)
        elif architecture == 'vgg':
            model = vgg16(weights=VGG16_Weights.DEFAULT)
        elif architecture == 'resnet':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        model.eval()
        model.to(self.device)
        return model

    def _register_hooks(self):
        # Register forward hooks to extract features from the last layer before classification
        if isinstance(self.model, models.AlexNet):
            # For AlexNet, features before the final classifier layer at index 5
            self.model.classifier[5].register_forward_hook(self._hook_fn)
        elif isinstance(self.model, models.VGG):
            # For VGG, features before the first classifier layer could be extracted
            # Here we use the output of the first classifier layer
            self.model.classifier[0].register_forward_hook(self._hook_fn)
        elif isinstance(self.model, models.ResNet):
            # For ResNet, use the avgpool layer
            self.model.avgpool.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features = output.detach().cpu()

    def _get_transforms(self, augment: bool):
        # Standard ImageNet normalization values
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            # More aggressive augmentations for transfer learning scenarios
            augmentation_transforms = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize
            ]
            return transforms.Compose(augmentation_transforms)
        else:
            # Validation or no-augmentation scenario
            base_transforms = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]
            return transforms.Compose(base_transforms)

    def extract(self, image):
        """
        Extract features from a single image (PIL) after applying transforms.
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        _ = self.model(x)
        return self.features
