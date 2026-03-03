import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_glass_reflection(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像文件")
    
    # 方法1: 基于传统图像处理的方法
    def method1_traditional(image):
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用自适应阈值分割倒影区域
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 修复被倒影覆盖的区域
        result = cv2.inpaint(image, binary, 3, cv2.INPAINT_TELEA)
        return result
    
    # 方法2: 基于频域滤波的方法
    def method2_frequency(image):
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 转换到频域
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # 创建高通滤波器
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols, 2), np.float32)
        r = 30  # 滤波器半径
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
        mask[mask_area] = 0
        
        # 应用滤波器
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        # 归一化
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        img_back = np.uint8(img_back)
        
        # 转换回彩色
        result = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
        return result
    
    # 应用两种方法
    result1 = method1_traditional(image.copy())
    result2 = method2_frequency(image.copy())
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
    plt.title('方法1: 传统图像处理')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
    plt.title('方法2: 频域滤波')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result1, result2

# 使用示例
if __name__ == "__main__":
    # 替换为您的图像路径
    image_path = "example.png"
    result1, result2 = remove_glass_reflection(image_path)
