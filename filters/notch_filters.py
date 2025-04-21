import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from math import exp, pow

class IdealNotchFilter:
    def __init__(self):
        pass
    
    def apply_filter(self, shape, d0=10, u_k=0, v_k=0):
        P, Q = shape
        H = np.ones((P, Q), dtype=np.float32)

        for u in range(P):
            for v in range(Q):
                D_uv = np.sqrt((u - P/2 + u_k)**2 + (v - Q/2 + v_k)**2)
                D_muv = np.sqrt((u - P/2 - u_k)**2 + (v - Q/2 - v_k)**2)

                if D_uv <= d0 or D_muv <= d0:
                    H[u, v] = 0
        return H

class ButterworthNotchFilter:
    def __init__(self):
        pass
    
    def apply_filter(self, shape, d0=10, u_k=0, v_k=0, n=2):
        P, Q = shape
        H = np.ones((P, Q), dtype=np.float32)

        for u in range(P):
            for v in range(Q):
                D_uv = np.sqrt((u - P/2 + u_k)**2 + (v - Q/2 + v_k)**2)
                D_muv = np.sqrt((u - P/2 - u_k)**2 + (v - Q/2 - v_k)**2)

                H[u, v] = 1 / (1 + (d0**2 / (D_uv * D_muv))**n)
        return H
    
class GaussianNotchFilter:
    def __init__(self):
        pass
    
    def apply_filter(self, shape, d0=10, u_k=0, v_k=0):
        P, Q = shape
        H = np.ones((P, Q), dtype=np.float32)

        for u in range(P):
            for v in range(Q):
                D_uv = np.sqrt((u - P/2 + u_k)**2 + (v - Q/2 + v_k)**2)
                D_muv = np.sqrt((u - P/2 - u_k)**2 + (v - Q/2 - v_k)**2)

                H[u, v] = 1 - np.exp(-0.5 * ((D_uv * D_muv) / (d0**2)) )

        return H
