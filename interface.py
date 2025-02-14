import os
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

# ------------------------------
# 定义残差块（参考 EDSR） / Define Residual Block (Reference: EDSR)
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

# ------------------------------
# 定义上采样模块（参考 ESPCN 的 PixelShuffle） / Define Upsample Block (Reference: PixelShuffle from ESPCN)
# ------------------------------
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.relu  = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.pixel_shuffle(self.conv(x)))

# ------------------------------
# 定义超分辨率模型 / Define Super Resolution Model
# ------------------------------
class SuperResolutionModel(nn.Module):
    def __init__(self, upscale_factor=4, num_channels=3, num_features=64, num_residuals=16):
        super(SuperResolutionModel, self).__init__()
        # 特征提取 / Feature Extraction
        self.conv_input = nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4)
        self.relu       = nn.ReLU(inplace=True)
        
        # 残差块 / Residual Blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_residuals)])
        
        # 残差块后卷积 + 全局跳跃连接 / Convolution after residual blocks + Global Skip Connection
        self.conv_mid = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # 上采样模块：当 upscale_factor==4 时，使用两个2倍上采样块
        # Upsample module: when upscale_factor==4, use two 2x upsample blocks
        upsample_layers = []
        if upscale_factor == 4:
            upsample_layers.append(UpsampleBlock(num_features, 2))
            upsample_layers.append(UpsampleBlock(num_features, 2))
        elif upscale_factor == 2:
            upsample_layers.append(UpsampleBlock(num_features, 2))
        else:
            upsample_layers.append(UpsampleBlock(num_features, upscale_factor))
        self.upsample = nn.Sequential(*upsample_layers)
        
        # 重构层 / Reconstruction Layer
        self.conv_output = nn.Conv2d(num_features, num_channels, kernel_size=9, padding=4)
    
    def forward(self, x):
        out1 = self.relu(self.conv_input(x))
        out  = self.residual_blocks(out1)
        out  = self.conv_mid(out)
        out  = out + out1  # 全局跳跃连接 / Global Skip Connection
        out  = self.upsample(out)
        out  = self.conv_output(out)
        return out

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 初始化模型，并加载 checkpoint 权重 / Initialize the model and load checkpoint weights
    model = SuperResolutionModel(
        upscale_factor=args.upscale_factor,
        num_channels=3,
        num_features=args.num_features,
        num_residuals=args.num_residuals
    ).to(device)
    
    if os.path.isfile(args.checkpoint):
        print("Loading checkpoint from:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Checkpoint file not found:", args.checkpoint)
        return
    
    model.eval()
    
    # 确保输出目录存在 / Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义转换操作 / Define transformation operations
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    # 获取输入目录中所有图片路径 / Get all image file paths from the input directory
    image_files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # 加载高分辨率图片，并转换为 RGB / Load high-resolution image and convert to RGB
        hr_image = Image.open(img_path).convert("RGB")
        # 根据 upscale_factor 生成低分辨率图片（bicubic 下采样） / Generate low-resolution image based on upscale_factor (bicubic downsampling)
        lr_size = (hr_image.width // args.upscale_factor, hr_image.height // args.upscale_factor)
        lr_image = hr_image.resize(lr_size, Image.BICUBIC)
        
        # 将低分辨率图片转换为 tensor，并添加 batch 维度 / Convert low-resolution image to tensor and add batch dimension
        lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 可选：使用混合精度加速推理 / Optional: use mixed precision for inference
            with torch.cuda.amp.autocast():
                sr_tensor = model(lr_tensor)
        # 将输出结果截断到 [0, 1] / Clamp output tensor to [0, 1]
        sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)
        # 去除 batch 维度，转换为 PIL 图片 / Remove batch dimension and convert to PIL image
        sr_image = to_pil(sr_tensor.squeeze(0).cpu())
        
        # 构造保存路径，文件名后添加 _SR / Construct output path and append _SR to the filename
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_SR.png")
        sr_image.save(output_path)
    
    print("Processing complete. Super-resolved images are saved in:", args.output_dir)

# nohup 命令示例 / Example nohup command:
# nohup python3 interface.py --checkpoint ./checkpoints/best_model.pth.tar --input_dir DIV2K_valid_HR --output_dir ./results --upscale_factor 4
# C:/Users/stran/anaconda3/envs/test/python.exe d:/super/interface.py --checkpoint best_model.pth.tar --input_dir DIV2K_valid_HR --output_dir ./results --upscale_factor 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Resolution Inference Interface")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Directory containing HR images (e.g., DIV2K_valid_HR)")
    parser.add_argument('--output_dir', type=str, default="./results",
                        help="Directory to save the super-resolved images")
    parser.add_argument('--upscale_factor', type=int, default=4,
                        help="Upscale factor for super resolution")
    parser.add_argument('--num_features', type=int, default=64,
                        help="Number of feature maps in the model")
    parser.add_argument('--num_residuals', type=int, default=16,
                        help="Number of residual blocks in the model")
    args = parser.parse_args()
    main(args)
