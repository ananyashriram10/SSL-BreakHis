from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def downstream_train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def simmim_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.01),
            transforms.ToTensor(),
        ]
    )


class TwoCropsTransform:
    def __init__(self, image_size: int = 224):
        self.base_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.25, 0.25, 0.2, 0.03)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)

