import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from math import exp, pow

class IdealNotchFilter:
    def __init__(self):
        pass
    
    def apply_filter(self, shape, d0=10, u_k=0, v_k=0):
        M, N = shape
        H = np.ones((M, N), dtype=np.float32)

        for u in range(M):
            for v in range(N):
                D_uv = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)
                D_muv = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)

                if D_uv <= d0 or D_muv <= d0:
                    H[u, v] = 0
        return H

class ButterworthNotchFilter:
    def __init__(self):
        pass
    
    def apply_filter(self, shape, d0=10, u_k=0, v_k=0, n=2):
        M, N = shape
        H = np.ones((M, N), dtype=np.float32)

        for u in range(M):
            for v in range(N):
                D_uv = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)
                D_muv = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)

                H[u, v] = (1 / (1 + (d0 / D_muv)**(2 * n))) * (1 / (1 + (d0 / D_uv)**(2 * n)))

        return H
    
class GaussianNotchFilter:
    def __init__(self):
        pass
    
    def apply_filter(self, shape, d0=10, u_k=0, v_k=0):
        M, N = shape
        H = np.ones((M, N), dtype=np.float32)

        for u in range(M):
            for v in range(N):
                D_uv = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)
                D_muv = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)

                H[u, v] = (1 - np.exp(-(D_uv**2) / (2 * d0**2))) * (1 - np.exp(-(D_muv**2) / (2 * d0**2)))

        return H
