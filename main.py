import tkinter as tk
from ctypes import windll
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import pathlib
import matplotlib.image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

def gaussian_notch_filter(shape, d0=10, u_k=0, v_k=0):
    P, Q = shape
    H = np.ones((P, Q), dtype=np.float32)

    for u in range(P):
        for v in range(Q):
            D_uv = np.sqrt((u - P/2 + u_k)**2 + (v - Q/2 + v_k)**2)
            D_muv = np.sqrt((u - P/2 - u_k)**2 + (v - Q/2 - v_k)**2)

            H[u, v] = 1 - np.exp(-0.5 * ((D_uv * D_muv) / (d0**2)) )

    return H

def ideal_notch_filter(shape, d0=10, u_k=0, v_k=0):
    P, Q = shape
    H = np.ones((P, Q), dtype=np.float32)

    for u in range(P):
        for v in range(Q):
            D_uv = np.sqrt((u - P/2 + u_k)**2 + (v - Q/2 + v_k)**2)
            D_muv = np.sqrt((u - P/2 - u_k)**2 + (v - Q/2 - v_k)**2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0
    return H

def butterworth_notch_filter(shape, d0=10, u_k=0, v_k=0, n=2):
    P, Q = shape
    H = np.ones((P, Q), dtype=np.float32)

    for u in range(P):
        for v in range(Q):
            D_uv = np.sqrt((u - P/2 + u_k)**2 + (v - Q/2 + v_k)**2)
            D_muv = np.sqrt((u - P/2 - u_k)**2 + (v - Q/2 - v_k)**2)

            H[u, v] = 1 / (1 + (d0**2 / (D_uv * D_muv))**n)
    return H


def calculate_w(g, eta, window_size=(20, 25)):
    a, b = window_size

    mean_g = uniform_filter(g, size=(a, b))
    mean_eta = uniform_filter(eta, size=(a, b))
    mean_g_eta = uniform_filter(g * eta, size=(a, b))
    mean_eta2 = uniform_filter(eta * eta, size=(a, b))

    numerator = mean_g_eta - mean_g * mean_eta
    denominator = mean_eta2 - mean_eta**2
    denominator = np.where(denominator == 0, 1e-5, denominator)

    w = numerator / denominator
    return w

class MainApp:
    def __init__(self):
        windll.shcore.SetProcessDpiAwareness(1)
        self.root = tk.Tk()
        # self.root.tk.call('tk', 'scaling', 1.5)
        self.root.resizable(True, True)
        self.root.title("Optimum Notch Filter")

        # === Tạo 6 Frame chứa ảnh ===
        self.frames = []

        
        # for i in range(6):
        #     frame = tk.LabelFrame(self.root, text=f"Cột {i+1}", bg="white")
        #     frame.grid(row=0, column=i, sticky="nsew", padx=5, pady=5)
        #     self.frames.append(frame)

        frame = tk.LabelFrame(self.root, text=f"Ảnh gốc g(x, y)", bg="white")
        frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frames.append(frame)

        frame = tk.LabelFrame(self.root, text=f"Phổ ảnh gốc G(u, v)", bg="white")
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

        # === Setup nội dung mẫu cho các frame ===
        # for frame in self.frames:
        #     label = tk.Label(frame, text="Hình ảnh ở đây", padx=80, pady=150)
        #     label.pack(expand=True, fill=tk.BOTH)

        self.anh_goc = tk.Label(self.frames[0], text="Ảnh gốc g(x, y)\nở đây",  padx=80, pady=150)
        self.anh_goc.pack(expand=True, fill=tk.BOTH)

        self.pho_anh_goc = tk.Label(self.frames[1], text="Phổ ảnh gốc\nG(u, v) ở đây",  padx=80, pady=150)
        self.pho_anh_goc.pack(expand=True, fill=tk.BOTH)

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

        # Số điểm
        points_frame = tk.Frame(self.control_frame, bg="lightgray")
        points_frame.pack(padx=600, fill=tk.X)

        tk.Label(points_frame, text="Số điểm:", bg="lightgray").grid(row=0, column=0, sticky='w', padx=5)
        self.number_of_points = tk.Entry(points_frame, justify='center', font=('Arial', 12))
        self.number_of_points.grid(row=0, column=1, sticky='ew', padx=5)
        self.number_of_points.insert(tk.END, '1')

        points_frame.grid_columnconfigure(1, weight=1)

        # Bán kính
        radius_frame = tk.Frame(self.control_frame, bg="lightgray")
        radius_frame.pack(padx=600, pady=5, fill=tk.X)

        tk.Label(radius_frame, text="Bán kính:", bg="lightgray").grid(row=0, column=0, sticky='w', padx=5)
        self.frequency = tk.Entry(radius_frame, justify='center', font=('Arial', 12))
        self.frequency.grid(row=0, column=1, sticky='ew', padx=5)
        self.frequency.insert(tk.END, '10')

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

    def browse_img(self):
        try:
            file_path = filedialog.askopenfilename(title="Load Image", filetypes=[('Images', ['*jpeg', '*png', '*jpg'])])
            if file_path:
                img = ImageOps.grayscale(Image.open(file_path))
                img.save(pathlib.Path("tmp/anh_goc.png")) # Lưu ảnh ban đầu vào tmp
                img = self.resize(img, 250)
                img_tk = ImageTk.PhotoImage(img)
                self.anh_goc.configure(image=img_tk, text="")
                self.anh_goc.image = img_tk
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
        img = np.asarray(Image.open(pathlib.Path("tmp/anh_goc.png")))
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        dft = 20 * np.log(np.abs(fshift) + 1)
        matplotlib.image.imsave(pathlib.Path("tmp/pho_anh.png"), dft, cmap="gray")
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
        img = matplotlib.image.imread('tmp/anh_goc.png')
        plt.imshow(img, cmap='gray')
        plt.title('Ảnh gốc g(x, y)')
        # plt.axis('off')

        plt.subplot(2, 3, 2)
        magnitude_spectrum = matplotlib.image.imread('tmp/pho_anh.png')
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
        fshift, dft = self.get_fshift_and_save_dft()
        plt.clf()
        plt.imshow(Image.open(pathlib.Path("tmp/pho_anh.png")), cmap="gray")
        self.set_plot_title("Nhấp vào hình ảnh để chọn điểm. (Nhấn phím bất kỳ để bắt đầu)")
        plt.waitforbuttonpress()
        self.set_plot_title(f"Click {self.number_of_points.get()} điểm bằng chuột")
        clicked_points = np.asarray(plt.ginput(int(self.number_of_points.get()), timeout=-1))
        print(clicked_points)
        plt.close()
        for i in range(len(clicked_points)):
            clicked_points[i][0], clicked_points[i][1] = clicked_points[i][1], clicked_points[i][0]
        if self.select_filter_var.get() in ["Gaussian", "Butterworth", "Ideal"]:
            img = np.asarray(Image.open(pathlib.Path("tmp/anh_goc.png")))
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)

            pho_anh = dft
            pho_anh = Image.fromarray(pho_anh).convert("L")
            pho_anh = self.resize(pho_anh, 250)
            pho_anh_tk = ImageTk.PhotoImage(pho_anh)
            self.pho_anh_goc.configure(image=pho_anh_tk, text="")
            self.pho_anh_goc.image = pho_anh_tk

            img_shape = img.shape
            H_total = np.ones(img_shape, dtype=np.float32)
            d0 = float(self.frequency.get())

            for (u_k, v_k) in clicked_points:
                u_shift = u_k - img_shape[0] // 2
                v_shift = v_k - img_shape[1] // 2
                if self.select_filter_var.get() == "Gaussian":
                    notch = gaussian_notch_filter(img_shape, d0, u_shift, v_shift)
                elif self.select_filter_var.get() == "Butterworth":
                    notch = butterworth_notch_filter(img_shape, d0, u_shift, v_shift, n=2)
                elif self.select_filter_var.get() == "Ideal":
                    notch = ideal_notch_filter(img_shape, d0, u_shift, v_shift)
                H_total *= notch

            G_shift = fshift
            HNP = H_total

            N_shift = G_shift * (1 - HNP)
            N = np.fft.ifftshift(N_shift)
            eta = np.fft.ifft2(N)
            eta = np.real(eta)

            w = calculate_w(img, eta, window_size=(20, 25))

            img = np.asarray(Image.open(pathlib.Path("tmp/anh_goc.png")))
            print("w", w)
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

if __name__ == "__main__":
    MainApp().run()
