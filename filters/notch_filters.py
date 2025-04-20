import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from math import exp, pow

class IdealNotchFilter:
    def __init__(self):
        pass
    
    def apply_filter(self, fshift, points, d0, path):
        M = fshift.shape[0]
        N = fshift.shape[1]
        print("M, N", M, N)
        H = np.ones((M, N), dtype = np.float32) # khởi tạo ma trận H với kích thước m x n
        for d in range(len(points)):
            uk = points[d][0] - N / 2  # tọa độ u của điểm cần lọc
            vk = points[d][1] - M / 2 # tọa độ v của điểm cần lọc
            vk, uk = uk, vk
            print("points", points)
            for u in range(M):
                for v in range(N):
                    # Get euclidean distance from point D(u,v) to the center
                    D1 = np.sqrt((u - M/2 - uk)**2 + (v - N/2 - vk)**2)
                    D2 = np.sqrt((u - M/2 + uk)**2 + (v - N/2 + vk)**2)
                    if D1 <= d0 or D2 <= d0:
                        H[u, v] = 0
            H *= H

        NotchFilter = H
        NotchRejectCenter = fshift * NotchFilter
        NotchReject = np.fft.ifftshift(NotchRejectCenter)
        inverse_NotchReject = np.fft.ifft2(NotchReject)
        inverse_NotchReject = np.abs(inverse_NotchReject)
        matplotlib.image.imsave(path, inverse_NotchReject, cmap = "gray")



        # img = cv2.imread('test/4.jpg', 0)

        # G = np.fft.fft2(img)
        # Gshift = np.fft.fftshift(G)

        # # Áp dụng Notch Pass Filter
        # HNP = 1 - H
        # NoiseCenter = Gshift * HNP
        # Noise = np.fft.ifftshift(NoiseCenter)
        # eta = np.fft.ifft2(Noise)
        # eta = np.abs(eta)

        # # Ước lượng hệ số w
        # w = 1  # đơn giản chọn w=1 (hoặc nhỏ hơn nếu cần)

        
        
        # def calculate_weight(img, eta):
        #     """
        #     Tính toán trọng số w cho Optimum Notch Filter.

        #     img: Ảnh gốc (g(x,y))
        #     eta: Nhiễu ước lượng (eta(x,y))

        #     Trả về trọng số w.
        #     """
        #     # Tính giá trị trung bình của g(x,y) và eta(x,y)
        #     mean_g = np.mean(img)
        #     mean_eta = np.mean(eta)
        #     mean_eta2 = np.mean(eta**2)

        #     # Tính tử số và mẫu số
        #     numerator = np.mean(img * eta) - mean_g * mean_eta
        #     denominator = mean_eta2 - mean_eta**2

        #     # Tránh chia cho 0
        #     if denominator == 0:
        #         w = 0
        #     else:
        #         w = numerator / denominator

        #     # Clamp w trong khoảng [0, 1]
        #     # w = np.clip(w, 0, 1)

        #     return w
        # calculate_weight(img, eta)
        # print("w", w)

        # # Khôi phục ảnh
        # f_hat = img - w * eta

        # # Chuẩn hóa kết quả về 0-255 để đẹp hơn
        # f_hat = np.clip(f_hat, 0, 255).astype(np.uint8)

        # # Hiển thị ảnh
        # plt.figure(figsize=(6,6))
        # plt.imshow(f_hat, cmap='gray')
        # plt.title('Ảnh phục hồi bằng Optimum Notch Filter')
        # plt.axis('off')
        # plt.show()

        return

class ButterworthNotchFilter:
    def __init__(self):
        pass
    
    def apply_filter(self, fshift, points, d0, path, order = 1):
        m = fshift.shape[0]
        n = fshift.shape[1]
        for u in range(m):
            for v in range(n):
                for d in range(len(points)):
                    u0 = points[d][0]
                    v0 = points[d][1]
                    u0, v0 = v0, u0
                    d1 = math.sqrt(pow(u - u0, 2) + pow(v - v0, 2))
                    d2 = math.sqrt(pow(u + u0, 2) + pow(v + v0, 2))
                    fshift[u][v] *= (1.0 / (1 + pow((d0 * d0) / (d1 * d2), order))) 
                    
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        matplotlib.image.imsave(path, img_back, cmap = "gray")
        return
    
class GaussianNotchFilter:
    def __init__(self):
        pass
    
    def apply_filter(self, fshift, points, d0, path):
        m = fshift.shape[0]
        n = fshift.shape[1]
        for u in range(m):
            for v in range(n):
                for d in range(len(points)):
                    u0 = points[d][0]
                    v0 = points[d][1]
                    u0, v0 = v0, u0
                    d1 = pow(pow(u - u0, 2) + pow(v - v0, 2), 0.5)
                    d2 = pow(pow(u + u0, 2) + pow(v + v0, 2), 0.5)
                    fshift[u][v] *= (1 - exp(-0.5 * (d1 * d2 / pow(d0, 2))))

        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        matplotlib.image.imsave(path, img_back, cmap = "gray")
        return
