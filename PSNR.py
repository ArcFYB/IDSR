from skimage import io, color
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 加载图片
imageA = io.imread('/home/fiko/Code/Super_Resolution/End2End_SR/experiments/Alsat_Ablation_attention/HCFNet/4919/300000_1_hr.png')
imageB = io.imread('/home/fiko/Code/Super_Resolution/End2End_SR/experiments/Alsat_Ablation_attention/HCFNet/4919/300000_1_sr.png')

# 转换RGB到灰度
def rgb2gray(rgb):
    return color.rgb2gray(rgb)

# 确保图片是灰度或RGB
if imageA.ndim == 3:
    imageA = rgb2gray(imageA)
if imageB.ndim == 3:
    imageB = rgb2gray(imageB)

# 确定数据范围
data_range = imageA.max() - imageA.min()

# 计算PSNR和SSIM
psnr_value = psnr(imageA, imageB, data_range=data_range)
ssim_value, _ = ssim(imageA, imageB, data_range=data_range, full=True)

print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")
