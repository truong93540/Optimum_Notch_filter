import tkinter as tk
from ctypes import windll
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import pathlib
import matplotlib.image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from filters.notch_filters import IdealNotchFilter, ButterworthNotchFilter, GaussianNotchFilter
import cv2
import time
from pathlib import Path
import scipy.ndimage

def calculate_w(g, eta, window_size=(9, 9)): 

    mean_g = cv2.blur(g, (window_size))
    mean_eta = cv2.blur(eta, (window_size))
    mean_g_eta = cv2.blur(g * eta, (window_size))
    mean_eta2 = cv2.blur(eta * eta, (window_size))

    numerator = mean_g_eta - mean_g * mean_eta
    denominator = mean_eta2 - mean_eta**2
    denominator = np.where(denominator == 0, 1e-5, denominator)

    w = numerator / denominator
    return w

def calculate_mse(original, restored):
    """
    Tính Mean Squared Error (MSE) giữa ảnh gốc và ảnh khôi phục
    MSE = (1/MN) * Σ(original - restored)²
    """
    # Chuyển đổi sang float để tránh tràn số
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    
    # Tính MSE
    mse = np.mean((original - restored) ** 2)
    return mse

def evaluate_restoration(original, restored):
    """
    Đánh giá chất lượng khôi phục ảnh
    """
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(restored, Image.Image):
        restored = np.array(restored)

    # Chuyển đổi sang float để tránh tràn số
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    
    # Tính MSE
    mse = np.mean((original - restored) ** 2)
    
    # Đánh giá chất lượng khôi phục (có thể dựa trên MSE hoặc các tiêu chí khác)
    if mse < 50:
        evaluation = "Tốt"
    elif mse < 100:
        evaluation = "Trung bình"
    else:
        evaluation = "Kém"
    
    return mse, evaluation

def  calculate_snr(original, noisy):
    """
    Tính SNR (Signal-to-Noise Ratio) giữa ảnh gốc và ảnh nhiễukhôi phục.
    SNR = 10 * log10 (Power_signal / Power_noise)
    """
    # Chuyển sang float để tránh tràn số
    original = original.astype(np.float64)
    noisy = noisy.astype(np.float64)
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - noisy) ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def auto_detect_noise_peaks(magnitude_spectrum, threshold_ratio=0.7, min_distance=10, center_exclude_radius=30):
    """
    Phát hiện các đốm sáng trên phổ tần số và loại bỏ các điểm gần tâm.
    """
    h, w = magnitude_spectrum.shape
    center = (h // 2, w // 2)
    # Chuẩn hóa phổ về [0, 1]
    norm_mag = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
    # Tìm local maxima
    neighborhood = scipy.ndimage.generate_binary_structure(2, 2)
    local_max = scipy.ndimage.maximum_filter(norm_mag, size=min_distance) == norm_mag
    # Lọc theo ngưỡng
    detected_peaks = np.where((local_max) & (norm_mag > threshold_ratio))
    # Loại bỏ các điểm gần tâm
    peaks = []
    for y, x in zip(detected_peaks[0], detected_peaks[1]):
        if np.sqrt((y - center[0])**2 + (x - center[1])**2) > center_exclude_radius:
            peaks.append((y, x))
    return peaks

class MainApp:
    def __init__(self):
        windll.shcore.SetProcessDpiAwareness(1)
        self.root = tk.Tk()
        self.root.resizable(True, True)
        # self.root.option_add("*Font", "Arial 12")
        self.root.title("Optimum Notch Filter")

        # === Tạo 6 Frame chứa ảnh
        self.frames = []

        frame = tk.LabelFrame(self.root, text=f"Ảnh nhiễu g(x, y)", bg="white")
        frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frames.append(frame)

        frame = tk.LabelFrame(self.root, text=f"Phổ ảnh nhiễu G(u, v)", bg="white")
        frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.frames.append(frame)

        frame = tk.LabelFrame(self.root, text=f"Nhiễu trong miền tần số N(u, v)", bg="white")
        frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.frames.append(frame)

        frame = tk.LabelFrame(self.root, text=f"Nhiễu trong miền không gian η(x, y)", bg="white")
        frame.grid(row=0, column=3, sticky="nsew", padx=5, pady=5)
        self.frames.append(frame)

        frame = tk.LabelFrame(self.root, text=f"Trọng số W(x, y)", bg="white")
        frame.grid(row=0, column=4, sticky="nsew", padx=5, pady=5)
        self.frames.append(frame)

        frame = tk.LabelFrame(self.root, text=f"Ảnh sau khôi phục", bg="white")
        frame.grid(row=0, column=5, sticky="nsew", padx=5, pady=5)
        self.frames.append(frame)

        # === Chia đều 6 cột ===
        for i in range(6):
            self.root.grid_columnconfigure(i, weight=1)

        self.anh_nhieu = tk.Label(self.frames[0], text="Ảnh nhiễu g(x, y)\nở đây",  padx=80, pady=150)
        self.anh_nhieu.pack(expand=True, fill=tk.BOTH)

        self.pho_anh_nhieu = tk.Label(self.frames[1], text="Phổ ảnh nhiễu\nG(u, v) ở đây",  padx=80, pady=150)
        self.pho_anh_nhieu.pack(expand=True, fill=tk.BOTH)

        self.nhieu_tan_so = tk.Label(self.frames[2], text="Nhiễu trong miền\ntần số N(u, v)\nở đây",  padx=80, pady=150)
        self.nhieu_tan_so.pack(expand=True, fill=tk.BOTH)

        self.nhieu_khong_gian = tk.Label(self.frames[3], text="Nhiễu trong miền\nkhông gian η(x, y)\nở đây",  padx=80, pady=150)
        self.nhieu_khong_gian.pack(expand=True, fill=tk.BOTH)

        self.trong_so = tk.Label(self.frames[4], text="Trọng số W(x, y)",  padx=80, pady=150)
        self.trong_so.pack(expand=True, fill=tk.BOTH)

        self.anh_khoi_phuc = tk.Label(self.frames[5], text="Ảnh sau khôi\nphục ở đây",  padx=80, pady=150)
        self.anh_khoi_phuc.pack(expand=True, fill=tk.BOTH)

        # === Frame chứa control nằm dưới 6 cột ===
        self.control_frame = tk.Frame(self.root, bg="lightgray")
        self.control_frame.grid(row=1, column=0, columnspan=6, sticky="nsew", pady=(10, 0))

        # Setup các control dọc
        self.setup_controls()

    def setup_controls(self):
        # Dùng frame con + grid để chữ trái, ô nhập/phím phải

        # Chọn bộ lọc
        filter_frame = tk.Frame(self.control_frame, bg="lightgray")
        filter_frame.pack(padx=600, fill=tk.X, pady=5)

        tk.Label(filter_frame, text="Chọn bộ lọc:", bg="lightgray").grid(row=0, column=0, sticky='w', padx=5)
        self.select_filter_var = tk.StringVar(value='Ideal')
        self.select_filter = tk.OptionMenu(filter_frame, self.select_filter_var, 'Ideal', 'Butterworth', 'Gaussian')
        self.select_filter.grid(row=0, column=1, sticky='ew', padx=5)

        filter_frame.grid_columnconfigure(1, weight=1)

        # Bán kính
        radius_frame = tk.Frame(self.control_frame, bg="lightgray")
        radius_frame.pack(padx=600, pady=5, fill=tk.X)

        tk.Label(radius_frame, text="Bán kính:", bg="lightgray").grid(row=0, column=0, sticky='w', padx=5)
        self.frequency = tk.Entry(radius_frame, justify='center', font=('Arial', 12))
        self.frequency.grid(row=0, column=1, sticky='ew', padx=5)
        self.frequency.insert(tk.END, '5')

        radius_frame.grid_columnconfigure(1, weight=1)

        # Các nút bấm
        self.btn_browse_img = tk.Button(self.control_frame, text="Chọn ảnh", bg="lightblue", command=self.browse_img)
        self.btn_browse_img.pack(padx=600, pady=5, fill=tk.X)

        self.btn_apply_filter = tk.Button(self.control_frame, text="Áp dụng lọc", bg="lightblue", command=self.apply_filter)
        self.btn_apply_filter.pack(padx=600, pady=5, fill=tk.X)

        self.btn_save_img = tk.Button(self.control_frame, text="Lưu ảnh", bg="lightblue", command=self.save_img)
        self.btn_save_img.pack(padx=600, pady=5, fill=tk.X)

        self.btn_summary = tk.Button(self.control_frame, text="Chi tiết ", bg="lightblue", command=self.detail)
        self.btn_summary.pack(padx=600, pady=5, fill=tk.X)

        self.btn_summary.pack(padx=600, pady=5, fill=tk.X)

        # Thêm label hiển thị kết quả đánh giá và thời gian xử lý
        self.result_label = tk.Label(self.control_frame, text="", bg="lightgray", fg="blue", font=("Arial", 12, "bold"))
        self.result_label.pack(padx=600, pady=5, fill=tk.X)
        self.time_label = tk.Label(self.control_frame, text="", bg="lightgray", fg="green", font=("Arial", 12))
        self.time_label.pack(padx=600, pady=5, fill=tk.X)

    def browse_img(self):
        try:
            file_path = filedialog.askopenfilename(title="Load Image", filetypes=[('Images', ['*jpeg', '*png', '*jpg'])])
            
            if file_path:
                img = ImageOps.grayscale(Image.open(file_path))
                img.save(pathlib.Path("tmp/anh_nhieu.png")) # Lưu ảnh ban đầu vào tmp
                img = self.resize(img, 250)
                img_tk = ImageTk.PhotoImage(img)
                self.anh_nhieu.configure(image=img_tk, text="")
                self.anh_nhieu.image = img_tk
                file_path = Path(file_path)
                path_anh_nhieu = file_path.parent.parent / 'anhgoc' / file_path.name
                print(path_anh_nhieu)
                anh_nhieu = ImageOps.grayscale(Image.open(path_anh_nhieu))
                anh_nhieu.save(pathlib.Path("tmp/anh_goc.png"))
        except Exception as e:
            messagebox.showerror("An error occurred!", str(e))

    def run(self):
        self.root.mainloop()

    def resize(self,img, width=200 ):
        fixed_width = width  # Chiều ngang mong muốn
        w_percent = (fixed_width / float(img.width))
        height_size = int((float(img.height) * float(w_percent)))
        img = img.resize((fixed_width, height_size), Image.LANCZOS)
        return img
    
    def get_fshift_and_save_dft(self):
        img = np.asarray(Image.open(pathlib.Path("tmp/anh_nhieu.png")))
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        dft = 20 * np.log(np.abs(fshift) + 1)
        matplotlib.image.imsave(pathlib.Path("tmp/pho_anh_nhieu.png"), dft, cmap="gray")
        return fshift, dft
    
    def set_plot_title(self, title, fs=16):
        plt.title(title, fontsize=fs)
    
    def save_img(self):
        try:
            save_path = filedialog.asksaveasfilename(title="Save Image", defaultextension=".png", filetypes=[('PNG files', '*.png'), ('JPEG files', '*.jpeg'), ('All files', '*.*')])
            if save_path:
                Image.open(pathlib.Path("tmp/anh_khoi_phuc.png")).save(save_path)
        except Exception as e:
            messagebox.showerror("An error occurred!", str(e))

    def detail(self):
        plt.figure(figsize=(10, 5))

        plt.subplot(2, 3, 1)
        img = matplotlib.image.imread('tmp/anh_nhieu.png')
        plt.imshow(img, cmap='gray')
        plt.title('Ảnh nhiễu g(x, y)')
        # plt.axis('off')

        plt.subplot(2, 3, 2)
        magnitude_spectrum = matplotlib.image.imread('tmp/pho_anh_nhieu.png')
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Phổ ảnh gốc G(u, v)')
        # plt.axis('off')

        plt.subplot(2, 3, 3)
        nhieu_tan_so = matplotlib.image.imread('tmp/nhieu_tan_so.png')
        plt.imshow(nhieu_tan_so, cmap='gray')
        plt.title('Nhiễu trong miền tần số N(u, v)')
        # plt.axis('off')

        plt.subplot(2, 3, 4)
        eta = matplotlib.image.imread('tmp/nhieu_khong_gian.png')
        plt.imshow(eta, cmap='gray')
        plt.title('Nhiễu trong miền không gian η(x,y)')
        # plt.axis('off')

        plt.subplot(2, 3, 5)
        w = matplotlib.image.imread('tmp/trong_so.png')
        plt.imshow(w, cmap='gray')
        plt.title('Trọng số w(x,y)')
        # plt.axis('off')

        plt.subplot(2, 3, 6)
        img_restored = matplotlib.image.imread('tmp/anh_khoi_phuc.png')
        plt.imshow(img_restored, cmap='gray')
        plt.title('Ảnh sau khôi phục')
        # plt.axis('off')

        plt.tight_layout()
        plt.show()

    def apply_filter(self):
        try:
            fshift, dft = self.get_fshift_and_save_dft()
            plt.clf()
            plt.imshow(Image.open(pathlib.Path("tmp/pho_anh_nhieu.png")), cmap="gray")
            self.set_plot_title("Nhấp vào hình ảnh để chọn điểm. (Nhấn phím bất kỳ để bắt đầu)")
            plt.waitforbuttonpress()
            self.set_plot_title("Nhấp chuột chọn điểm (nhấn Enter để kết thúc)")
            clicked_points = np.asarray(plt.ginput(n=-1, timeout=0))
            plt.close()
            
            for i in range(len(clicked_points)):
                clicked_points[i][0], clicked_points[i][1] = clicked_points[i][1], clicked_points[i][0]
            
            if self.select_filter_var.get() in ["Gaussian", "Butterworth", "Ideal"]:
                start_time = time.time()
                img = np.asarray(Image.open(pathlib.Path("tmp/anh_nhieu.png")))
                f = np.fft.fft2(img)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)

                pho_anh_nhieu = dft
                pho_anh_nhieu = Image.fromarray(pho_anh_nhieu).convert("L")
                pho_anh_nhieu = self.resize(pho_anh_nhieu, 250)
                pho_anh_nhieu_tk = ImageTk.PhotoImage(pho_anh_nhieu)
                self.pho_anh_nhieu.configure(image=pho_anh_nhieu_tk, text="")
                self.pho_anh_nhieu.image = pho_anh_nhieu_tk

                img_shape = img.shape
                H_total = np.ones(img_shape, dtype=np.float32)
                d0 = float(self.frequency.get())

                for (u_k, v_k) in clicked_points:
                    u_shift = u_k - img_shape[0] // 2
                    v_shift = v_k - img_shape[1] // 2
                    if self.select_filter_var.get() == "Gaussian":
                        notch = GaussianNotchFilter().apply_filter(img_shape, d0, u_shift, v_shift)
                    elif self.select_filter_var.get() == "Butterworth":
                        notch = ButterworthNotchFilter().apply_filter(img_shape, d0, u_shift, v_shift, n=2)
                    elif self.select_filter_var.get() == "Ideal":
                        notch = IdealNotchFilter().apply_filter(img_shape, d0, u_shift, v_shift)
                    H_total *= notch

                G_shift = fshift
                HNP = H_total

                N_shift = G_shift * (1 - HNP)
                N = np.fft.ifftshift(N_shift)
                eta = np.fft.ifft2(N)
                eta = np.real(eta)

                w = calculate_w(img, eta, window_size=(20, 25))

                img = np.asarray(Image.open(pathlib.Path("tmp/anh_nhieu.png")))
                anhGoc = np.asarray(Image.open(pathlib.Path("tmp/anh_goc.png")))
                img_restored = img - w * eta

                # --- Update các ảnh trên giao diện ---
                matplotlib.image.imsave(pathlib.Path("tmp/nhieu_tan_so.png"), magnitude_spectrum * H_total, cmap="gray")
                nhieu_tan_so = magnitude_spectrum * H_total
                nhieu_tan_so = Image.fromarray(nhieu_tan_so).convert("L")
                nhieu_tan_so = self.resize(nhieu_tan_so, 250)
                nhieu_tan_so_tk = ImageTk.PhotoImage(nhieu_tan_so)
                self.nhieu_tan_so.configure(image=nhieu_tan_so_tk, text="")
                self.nhieu_tan_so.image = nhieu_tan_so_tk

                matplotlib.image.imsave(pathlib.Path("tmp/nhieu_khong_gian.png"), eta, cmap="gray")
                eta_norm = (eta - eta.min()) / (eta.max() - eta.min()) * 255
                eta_norm = eta_norm.astype(np.uint8)
                nhieu_khong_gian = Image.fromarray(eta_norm).convert("L")
                nhieu_khong_gian = self.resize(nhieu_khong_gian, 250)
                nhieu_khong_gian_tk = ImageTk.PhotoImage(nhieu_khong_gian)
                self.nhieu_khong_gian.configure(image=nhieu_khong_gian_tk, text="")
                self.nhieu_khong_gian.image = nhieu_khong_gian_tk

                matplotlib.image.imsave(pathlib.Path("tmp/trong_so.png"), w, cmap="gray")
                w_norm = (w - w.min()) / (w.max() - w.min()) * 255
                w_norm = w_norm.astype(np.uint8)
                trong_so = Image.fromarray(w_norm).convert("L")
                trong_so = self.resize(trong_so, 250)
                trong_so_tk = ImageTk.PhotoImage(trong_so)
                self.trong_so.configure(image=trong_so_tk, text="")
                self.trong_so.image = trong_so_tk

                matplotlib.image.imsave(pathlib.Path("tmp/anh_khoi_phuc.png"), img_restored, cmap="gray")
                anh_khoi_phuc = np.clip(img_restored, 0, 255).astype(np.uint8)
                anh_khoi_phuc = Image.fromarray(anh_khoi_phuc).convert("L")
                anh_khoi_phuc = self.resize(anh_khoi_phuc, 250)
                anh_khoi_phuc_tk = ImageTk.PhotoImage(anh_khoi_phuc)
                self.anh_khoi_phuc.configure(image=anh_khoi_phuc_tk, text="")
                self.anh_khoi_phuc.image = anh_khoi_phuc_tk

               # Tính MSE và đánh giá chất lượng khôi phục
                mse, evaluation = evaluate_restoration(anhGoc, img_restored)

                snr = calculate_snr(anhGoc, img_restored)
                
                # Hiển thị kết quả đánh giá và thời gian xử lý trên label
                evaluation_message = f"MSE: {mse:.2f} | SNR: {snr:.2f} dB | Đánh giá: {evaluation}"
                self.result_label.config(text=evaluation_message)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.time_label.config(text=f"Đã xử lý ảnh xong trong {elapsed_time:.2f} giây.")
            
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

if __name__ == "__main__":
    MainApp().run()