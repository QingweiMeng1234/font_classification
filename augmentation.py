from torchvision import transforms
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# class AddArtificialShadows(torch.nn.Module):
#     def __init__(self, shadow_depth=20, opacity=200):
#         super(AddArtificialShadows, self).__init__()
#         self.shadow_depth = shadow_depth  # Depth of the shadow in pixels
#         self.opacity = opacity  # Opacity of the shadow

#     def forward(self, img):
#         width, height = img.size
#         draw = ImageDraw.Draw(img, "RGBA")

#         # Add shadows
#         draw.rectangle([0, 0, width, self.shadow_depth], fill=(0, 0, 0, self.opacity))
#         draw.rectangle([0, height - self.shadow_depth, width, height], fill=(0, 0, 0, self.opacity))
#         draw.rectangle([0, 0, self.shadow_depth, height], fill=(0, 0, 0, self.opacity))
#         draw.rectangle([width - self.shadow_depth, 0, width, height], fill=(0, 0, 0, self.opacity))

#         return img

# class BlurEdges(torch.nn.Module):
#     def __init__(self, blur_radius=15, central_fraction=0.75):
#         super(BlurEdges, self).__init__()
#         self.blur_radius = blur_radius
#         self.central_fraction = central_fraction

#     def forward(self, img):
#         width, height = img.size
#         central_width = int(width * self.central_fraction)
#         central_height = int(height * self.central_fraction)

#         # Mask to keep center sharp
#         mask = Image.new("L", img.size, 0)
#         draw = ImageDraw.Draw(mask)
#         draw.rectangle([
#             (width - central_width) // 2,
#             (height - central_height) // 2,
#             (width + central_width) // 2,
#             (height + central_height) // 2
#         ], fill=255)

#         blurred_img = img.filter(ImageFilter.GaussianBlur(self.blur_radius))
#         final_img = Image.composite(img, blurred_img, mask)
#         return final_img

# class EnhanceCenter(torch.nn.Module):
#     def __init__(self, enhancement_factor=1.5, central_fraction=0.75):
#         super(EnhanceCenter, self).__init__()
#         self.enhancement_factor = enhancement_factor
#         self.central_fraction = central_fraction

#     def forward(self, img):
#         width, height = img.size
#         central_width = int(width * self.central_fraction)
#         central_height = int(height * self.central_fraction)

#         # Mask to enhance the center
#         mask = Image.new("L", img.size, "black")
#         draw = ImageDraw.Draw(mask)
#         draw.rectangle([
#             (width - central_width) // 2,
#             (height - central_height) // 2,
#             (width + central_width) // 2,
#             (height + central_height) // 2
#         ], fill="white")

#         enhancer = ImageEnhance.Brightness(img)
#         img_enhanced = enhancer.enhance(self.enhancement_factor)
#         final_img = Image.composite(img_enhanced, img, mask)
#         return final_img

# def augs():
#     transform_preprocessing = transforms.Compose([
#         transforms.Resize((128, 128)),
#         # transforms.Lambda(lambda img: AddArtificialShadows(shadow_depth=15, opacity=300)(img)),
#         # transforms.Lambda(lambda img: BlurEdges(blur_radius=3, central_fraction=0.9)(img)),
#         transforms.Lambda(lambda img: EnhanceCenter(enhancement_factor=1.2, central_fraction=0.6)(img)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(20),
#         # transforms.ColorJitter(brightness=0.5, contrast=0.5),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     print(type(transform_preprocessing))
#     return transform_preprocessing

# from torchvision import transforms
# import torch
# from PIL import Image
# import cv2
# import numpy as np

# class EdgeDetectionTransform:
#     def __init__(self, lower_threshold=100, upper_threshold=200):
#         self.lower_threshold = lower_threshold
#         self.upper_threshold = upper_threshold

#     def __call__(self, img):
#         # Convert PIL image to a numpy array
#         img_np = np.array(img.convert('L'))  # Convert to grayscale
#         # Apply Canny Edge Detector
#         edges = cv2.Canny(img_np, self.lower_threshold, self.upper_threshold)
#         # Convert numpy array back to PIL image
#         return Image.fromarray(edges)

# def augs():
#     return transforms.Compose([
#         transforms.Resize((128, 128)),  # Resize the image if necessary
#         EdgeDetectionTransform(100, 200),  # Apply edge detection
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(20),
#         transforms.ToTensor(),  # Convert the image to a tensor
#         transforms.Normalize((0.5,), (0.5,))  # Normalize (adjust depending on your needs)
#     ])

def augs():
    return transforms.Compose([
        transforms.Resize(256),  # Resize first to a fixed size
        transforms.RandomResizedCrop(224),  # Random crop to final size
        transforms.RandomHorizontalFlip(),  # Horizontal flip
        transforms.RandomRotation(20),  # Rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.ToTensor(),  # Convert to tensor BEFORE applying tensor-specific transforms
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])


