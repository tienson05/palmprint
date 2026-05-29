from torchvision import transforms

from src.transforms.clahe import CLAHETransform
from src.transforms.sharpen import SharpenTransform

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.98, 1.02)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    SharpenTransform(p=0.5),
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])