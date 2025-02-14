import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pathlib import Path

def calculate_metrics(img1_path, img2_path):
    """
    计算两张图片的PSNR和SSIM值
    Calculate the PSNR and SSIM values for two images.
    
    参数 / Parameters:
    img1_path: 第一张图片的路径 / Path to the first image.
    img2_path: 第二张图片的路径 / Path to the second image.
    
    返回 / Returns:
    tuple: (psnr_value, ssim_value) 如果计算成功，否则返回 None
           (psnr_value, ssim_value) if successful, otherwise None.
    """
    try:
        # 读取图片 / Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # 确保两张图片尺寸相同 / Ensure both images have the same dimensions
        if img1.shape != img2.shape:
            print(f"图片尺寸不匹配 / Image size mismatch: {img1_path} vs {img2_path}")
            return None
        
        # 计算PSNR / Calculate PSNR
        psnr_value = psnr(img1, img2)
        
        # 计算SSIM / Calculate SSIM
        # 转换为灰度图进行SSIM计算 / Convert images to grayscale for SSIM calculation
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(img1_gray, img2_gray)
        
        return psnr_value, ssim_value
    
    except Exception as e:
        print(f"处理图片时出错 / Error processing images: {img1_path} vs {img2_path}")
        print(f"错误信息 / Error message: {str(e)}")
        return None

def compare_folders(folder1, folder2):
    """
    比较两个文件夹中的PNG图片，文件名前四个数字相同的视为对应图片
    Compare PNG images in two folders; images with the same first 4 digits in their filename are considered corresponding.
    
    参数 / Parameters:
    folder1: 第一个文件夹路径 / Path to the first folder.
    folder2: 第二个文件夹路径 / Path to the second folder.
    """
    # 获取所有PNG文件并按前4个数字分组 / Get all PNG files and group them by the first 4 digits of the filename
    folder1_files = {}
    folder2_files = {}
    
    # 处理第一个文件夹 / Process the first folder
    for f in Path(folder1).glob("*.png"):
        if f.name[:4].isdigit():
            folder1_files[f.name[:4]] = f
    
    # 处理第二个文件夹 / Process the second folder
    for f in Path(folder2).glob("*.png"):
        if f.name[:4].isdigit():
            folder2_files[f.name[:4]] = f
    
    # 找出共同的前缀 / Find common prefixes
    common_prefixes = set(folder1_files.keys()) & set(folder2_files.keys())
    
    if not common_prefixes:
        print("没有找到可以比较的PNG文件 / No comparable PNG files found")
        return
    
    # 存储所有PSNR和SSIM值用于计算平均值 / Store all PSNR and SSIM values for computing averages
    all_psnr = []
    all_ssim = []
    
    # 比较图片 / Compare images
    for prefix in sorted(common_prefixes):
        img1_path = folder1_files[prefix]
        img2_path = folder2_files[prefix]
        
        # 可选：打印比较图片对的信息 / Optional: print information about the image pair being compared
        # print(f"\n比较图片对 / Comparing image pair: {img1_path.name} <-> {img2_path.name}")
        
        metrics = calculate_metrics(str(img1_path), str(img2_path))
        
        if metrics:
            psnr_value, ssim_value = metrics
            all_psnr.append(psnr_value)
            all_ssim.append(ssim_value)
            # 可选：打印单张图片的PSNR和SSIM值 / Optional: print the PSNR and SSIM values for the pair
            # print(f"PSNR: {psnr_value:.2f}")
            # print(f"SSIM: {ssim_value:.4f}")
    
    if all_psnr and all_ssim:
        print(f"\n处理完成! 共比较了 {len(all_psnr)} 对图片 / Processing complete! Compared {len(all_psnr)} pairs of images")
        print(f"平均PSNR / Average PSNR: {np.mean(all_psnr):.4f}")
        print(f"平均SSIM / Average SSIM: {np.mean(all_ssim):.4f}")
    else:
        print("没有成功比较任何图片 / No images were successfully compared")


# 平均PSNR: 27.5790 / Average PSNR: 27.5790
# 平均SSIM: 0.8031 / Average SSIM: 0.8031

if __name__ == "__main__":
    # 示例使用 / Example usage:
    folder1 = r"D:\super\DIV2K_valid_HR"
    folder2 = r"D:\super\results"
    
    compare_folders(folder1, folder2)
