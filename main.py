import pathlib
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from filters.notch_filters import IdealNotchFilter, ButterworthNotchFilter, GaussianNotchFilter
from ctypes import windll

if not os.path.exists('tmp'):
    os.makedirs('tmp')

def set_plot_title(title, fs=16):
    plt.title(title, fontsize=fs)

def get_local_mean(image, i, j, filter_size):
    half_size = filter_size // 2
    h, w = image.shape
    top = max(i - half_size, 0)
    bottom = min(i + half_size + 1, h)
    left = max(j - half_size, 0)
    right = min(j + half_size + 1, w)
    local_patch = image[top:bottom, left:right]
    return np.mean(local_patch)

def apply_optimum_notch_filter(image, filter_size):
    h, w = image.shape

    # Biến đổi Fourier và lấy noise map
    ft_noisy = np.fft.fft2(image)
    ft_noisy_shift = np.fft.fftshift(ft_noisy)
    noise_map = np.abs(ft_noisy_shift)

    noise_map_ifft = np.fft.ifftshift(noise_map)
    noise_map_ifft = np.fft.ifft2(noise_map_ifft)
    noise_map_ifft_abs = np.abs(noise_map_ifft)

    image_mul_noise = image * noise_map_ifft_abs
    noise_mul_noise = noise_map_ifft_abs * noise_map_ifft_abs

    final_image = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            local_mean_image_mul_noise = get_local_mean(image_mul_noise, i, j, filter_size)
            local_mean_image = get_local_mean(image, i, j, filter_size)
            local_mean_noise_map = get_local_mean(noise_map_ifft_abs, i, j, filter_size)
            local_mean_noise_mul_noise = get_local_mean(noise_mul_noise, i, j, filter_size)

            numerator = local_mean_image_mul_noise - local_mean_image * local_mean_noise_map
            denominator = local_mean_noise_mul_noise - (local_mean_noise_map ** 2)

            if denominator == 0:
                w_opt = 0
            else:
                w_opt = numerator / denominator

            final_value = image[i, j] - w_opt * noise_map_ifft_abs[i, j]
            final_image[i, j] = np.clip(final_value, 0, 255)

    return final_image

class MainApp:
    def __init__(self):
        windll.shcore.SetProcessDpiAwareness(1)
        self.root = tk.Tk()
        self.root.tk.call('tk', 'scaling', 1.5)
        self.root.resizable(0, 0)
        self.root.title("Notch Filter")
        self.root.iconphoto(False, tk.PhotoImage(file=pathlib.Path("imgs/icon.png")))

        self.left_frame = tk.LabelFrame(self.root, text="Original Image")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.right_frame = tk.LabelFrame(self.root, text="Filtered Image")
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        for col in range(2):
            self.left_frame.columnconfigure(col, weight=1)
        for row in range(6):
            self.left_frame.rowconfigure(row, weight=1)

        self.setup_left_frame()
        self.setup_right_frame()

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def setup_left_frame(self):
        self.original_img = tk.Label(self.left_frame, text="Load an image \n to preview it here!", padx=150, pady=150)
        self.original_img.grid(row=0, column=0, columnspan=2)

        self.btn_browse_img = tk.Button(self.left_frame, text="Browse Image", bg="lightblue", command=self.browse_img)
        self.btn_browse_img.grid(row=1, column=0, sticky="nsew")

        self.btn_apply_filter = tk.Button(self.left_frame, text="Apply Filter", bg="lightblue", command=self.apply_filter)
        self.btn_apply_filter.grid(row=1, column=1, sticky="nsew")

        tk.Label(self.left_frame, text="Select type of Notch Filter: ").grid(row=2, column=0, sticky='nsew')
        self.select_filter_var = tk.StringVar(value='Ideal')
        self.select_filter = tk.OptionMenu(self.left_frame, self.select_filter_var, 'Ideal', 'Butterworth', 'Gaussian', 'Optimum')
        self.select_filter.grid(row=2, column=1, sticky='nsew')

        tk.Label(self.left_frame, text="Number of Points: ").grid(row=3, column=0, sticky='nsew')
        self.number_of_points = tk.Entry(self.left_frame)
        self.number_of_points.grid(row=3, column=1, sticky='nsew')
        self.number_of_points.insert(tk.END, '1')

        tk.Label(self.left_frame, text="Band Radius: ").grid(row=4, column=0, sticky='nsew')
        self.frequency = tk.Entry(self.left_frame)
        self.frequency.grid(row=4, column=1, sticky='nsew')
        self.frequency.insert(tk.END, '5')

        tk.Label(self.left_frame, text="Order of \n Butterworth Filter").grid(row=5, column=0, sticky='nsew')
        self.butterworth_order = tk.Entry(self.left_frame)
        self.butterworth_order.grid(row=5, column=1, sticky='nsew')
        self.butterworth_order.insert(tk.END, '1')

    def setup_right_frame(self):
        self.filter_img = tk.Label(self.right_frame, text="Apply filter to an image\nto view it here!", padx=150, pady=150)
        self.filter_img.pack()

        self.btn_save_img = tk.Button(self.right_frame, text="Save this Image", bg="lightblue", command=self.save_img)
        self.btn_save_img.pack(fill=tk.X)

        self.btn_summary = tk.Button(self.right_frame, text="Show Summary", bg="lightblue", command=self.show_summary)
        self.btn_summary.pack(fill=tk.X)

        self.info_lbl = tk.Label(self.right_frame, text="READY")
        self.info_lbl.pack(fill=tk.BOTH, side=tk.LEFT)

    def browse_img(self):
        try:
            file_path = filedialog.askopenfilename(title="Load Image", filetypes=[('Images', ['*jpeg', '*png', '*jpg'])])
            if file_path:
                img = ImageOps.grayscale(Image.open(file_path))
                img.save(pathlib.Path("tmp/original_img.png"))
                img_tk = ImageTk.PhotoImage(img)
                self.original_img.configure(image=img_tk, text="")
                self.original_img.image = img_tk
        except Exception as e:
            messagebox.showerror("An error occurred!", str(e))

    def get_fshift_and_save_dft(self):
        img = np.asarray(Image.open(pathlib.Path("tmp/original_img.png")))
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        dft = 20 * np.log(np.abs(fshift) + 1)
        matplotlib.image.imsave(pathlib.Path("tmp/dft.png"), dft, cmap="gray")
        return fshift, dft

    def apply_filter(self):
        try:
            self.info_lbl.configure(text="BUSY")
            fshift, dft = self.get_fshift_and_save_dft()

            if self.select_filter_var.get() == "Optimum":
                img = np.asarray(Image.open(pathlib.Path("tmp/original_img.png"))).astype(np.float32)
                filter_size = 200  # hoặc cho user nhập cũng được
                final_img = apply_optimum_notch_filter(img, filter_size)
                final_img = final_img.astype(np.uint8)
                Image.fromarray(final_img).save(pathlib.Path("tmp/filtered_img.png"))
            else:
                plt.clf()
                plt.imshow(Image.open(pathlib.Path("tmp/dft.png")), cmap="gray")
                set_plot_title("Click on image to choose points. (Press any key to start)")
                plt.waitforbuttonpress()
                set_plot_title(f"Select {self.number_of_points.get()} points with mouse click")
                points = np.asarray(plt.ginput(int(self.number_of_points.get()), timeout=-1))
                plt.close()

                radius = float(self.frequency.get())
                if self.select_filter_var.get() == "Ideal":
                    IdealNotchFilter().apply_filter(fshift, points, radius, pathlib.Path("tmp/filtered_img.png"))
                elif self.select_filter_var.get() == "Butterworth":
                    order = int(self.butterworth_order.get())
                    ButterworthNotchFilter().apply_filter(fshift, points, radius, pathlib.Path("tmp/filtered_img.png"), order)
                elif self.select_filter_var.get() == "Gaussian":
                    GaussianNotchFilter().apply_filter(fshift, points, radius, pathlib.Path("tmp/filtered_img.png"))

            self.update_filtered_image()
            self.info_lbl.configure(text="READY")
        except Exception as e:
            messagebox.showerror("An error occurred!", str(e))

    def update_filtered_image(self):
        img = ImageTk.PhotoImage(ImageOps.grayscale(Image.open(pathlib.Path("tmp/filtered_img.png"))))
        self.filter_img.configure(image=img)
        self.filter_img.image = img

    def save_img(self):
        try:
            save_path = filedialog.asksaveasfilename(title="Save Image", defaultextension=".png", filetypes=[('PNG files', '*.png'), ('JPEG files', '*.jpeg'), ('All files', '*.*')])
            if save_path:
                Image.open(pathlib.Path("tmp/filtered_img.png")).save(save_path)
        except Exception as e:
            messagebox.showerror("An error occurred!", str(e))

    def save_dft(self, img_path, save_path):
        img = np.asarray(ImageOps.grayscale(Image.open(img_path)))
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        dft = 20 * np.log(np.abs(fshift) + 1)
        matplotlib.image.imsave(save_path, dft, cmap="gray")

    def show_summary(self):
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(Image.open(pathlib.Path("tmp/original_img.png")), cmap="gray")
        axarr[0, 1].imshow(Image.open(pathlib.Path("tmp/filtered_img.png")), cmap="gray")
        axarr[1, 0].imshow(Image.open(pathlib.Path("tmp/dft.png")), cmap="gray")
        self.save_dft(pathlib.Path("tmp/filtered_img.png"), pathlib.Path("tmp/tdft.png"))
        axarr[1, 1].imshow(Image.open(pathlib.Path("tmp/tdft.png")), cmap="gray")
        plt.show()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    MainApp().run()
