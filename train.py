import os
import argparse
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
# 定义综合的超分辨率模型 / Define the Comprehensive Super Resolution Model
# ------------------------------
class SuperResolutionModel(nn.Module):
    def __init__(self, upscale_factor=4, num_channels=3, num_features=64, num_residuals=16):
        super(SuperResolutionModel, self).__init__()
        # 特征提取 / Feature extraction
        self.conv_input = nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4)
        self.relu       = nn.ReLU(inplace=True)
        
        # 多个残差块 / Multiple residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_residuals)])
        
        # 残差块后卷积，再与初始特征做全局跳跃连接 / Convolution after residual blocks and global skip connection with initial features
        self.conv_mid = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # 上采样模块：若 upscale_factor==4，则使用两个2倍上采样模块 / Upsample module: if upscale_factor == 4, use two 2x upsample blocks
        upsample_layers = []
        if upscale_factor == 4:
            upsample_layers.append(UpsampleBlock(num_features, 2))
            upsample_layers.append(UpsampleBlock(num_features, 2))
        elif upscale_factor == 2:
            upsample_layers.append(UpsampleBlock(num_features, 2))
        else:
            upsample_layers.append(UpsampleBlock(num_features, upscale_factor))
        self.upsample = nn.Sequential(*upsample_layers)
        
        # 重构层 / Reconstruction layer
        self.conv_output = nn.Conv2d(num_features, num_channels, kernel_size=9, padding=4)
    
    def forward(self, x):
        out1 = self.relu(self.conv_input(x))
        out  = self.residual_blocks(out1)
        out  = self.conv_mid(out)
        out  = out + out1  # 全局跳跃连接 / Global skip connection
        out  = self.upsample(out)
        out  = self.conv_output(out)
        return out

# ------------------------------
# 定义 Div2K 数据集（从 HR 生成 LR） / Define Div2K Dataset (Generate LR images from HR images)
# ------------------------------
class Div2KDataset(Dataset):
    def __init__(self, root_dir, scale_factor=4, crop_size=(500, 500), transform=None):
        """
        Args:
            root_dir (str): 存放高分辨率图像的文件夹路径。/ Folder path containing high-resolution (HR) images.
            scale_factor (int): 下采样比例。/ Down-sampling factor.
            crop_size (tuple): 裁剪后的高分辨率图像尺寸。/ Crop size for the HR images.
            transform: 对 HR 图像的预处理（默认转换为 tensor）。/ Preprocessing for HR images (default converts to tensor).
        """
        self.root_dir = root_dir
        self.image_files = sorted([
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.crop_transform = transforms.RandomCrop(crop_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_files[idx]).convert('RGB')
        # 裁剪出固定尺寸的 patch / Crop a patch with a fixed size
        hr_image = self.crop_transform(hr_image)
        hr = self.transform(hr_image)
        # 根据 scale_factor 得到低分辨率图像尺寸 / Calculate LR image size based on scale_factor
        lr_size = (hr_image.size[0] // self.scale_factor, hr_image.size[1] // self.scale_factor)
        lr_image = hr_image.resize(lr_size, Image.BICUBIC)
        lr = transforms.ToTensor()(lr_image)
        return lr, hr

# ------------------------------
# 保存 checkpoint 的函数 / Function to save a checkpoint
# ------------------------------
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# ------------------------------
# 主训练函数 / Main training function
# ------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # 构建模型并加载到设备 / Build the model and move it to the device
    model = SuperResolutionModel(
        upscale_factor=args.upscale_factor,
        num_channels=3,
        num_features=args.num_features,
        num_residuals=args.num_residuals
    ).to(device)
    
    criterion = nn.L1Loss()  # 像素级 L1 损失 / Pixel-level L1 loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Div2KDataset(root_dir=args.data_dir, scale_factor=args.upscale_factor, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    
    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0  # 用于统计连续未提升的 epoch 数 / Counter for epochs without improvement

    # 恢复训练（如有）/ Resume training if a checkpoint exists
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint:", args.resume)
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Checkpoint loaded from epoch", checkpoint['epoch'])
        else:
            print("No checkpoint found at", args.resume)
    
    # 训练循环（使用 tqdm 显示进度条）/ Training loop (using tqdm for progress bar)
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for i, (lr, hr) in enumerate(dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)
        pbar.close()
        
        epoch_loss /= len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Average Loss: {epoch_loss:.4f}  Time: {elapsed:.2f}s")
        
        scheduler.step()
        
        # 检查当前 epoch 是否有提升 / Check if the current epoch shows improvement
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0  # 重置计数器 / Reset counter
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth.tar')
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            save_checkpoint(checkpoint, filename=best_model_path)
            print("Best model saved at", best_model_path)
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")
        
        # 保存当前 epoch 的 checkpoint / Save checkpoint for the current epoch
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth.tar')
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        save_checkpoint(checkpoint, filename=checkpoint_path)
        print("Checkpoint saved at", checkpoint_path)
        
        # 如果连续若干个 epoch 没有提升，则触发早停 / Trigger early stopping if there is no improvement for several epochs
        if patience_counter >= args.patience:
            print("Early stopping triggered after {} epochs without improvement.".format(args.patience))
            break

# nohup 命令示例 / Example nohup command:
# nohup python3 /home/ubuntu/super/train.py --data_dir DIV2K_train_HR > train.log 2>&1 &
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Super Resolution Training with Progress Bar and Early Stopping")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Div2K folder containing HR images')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--upscale_factor', type=int, default=4, help='Upscale factor')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--step_size', type=int, default=50, help='Step size for LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for LR scheduler')
    parser.add_argument('--resume', type=str, default='', help='Path to resume checkpoint if needed')
    parser.add_argument('--num_features', type=int, default=64, help='Number of feature maps')
    parser.add_argument('--num_residuals', type=int, default=16, help='Number of residual blocks')
    parser.add_argument('--patience', type=int, default=100, help='Number of epochs with no improvement before early stopping')
    args = parser.parse_args()
    main(args)
