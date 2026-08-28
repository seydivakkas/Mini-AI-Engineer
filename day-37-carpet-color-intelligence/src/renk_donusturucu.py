"""
RGB <-> CIELAB Renk Uzayı Dönüştürücü (Perceptually Uniform Color Space Converter).
"""

import numpy as np


# D65 Standart Aydınlatıcı Referans Beyaz Noktası
X_N = 0.95047
Y_N = 1.00000
Z_N = 1.08883
DELTA = 6.0 / 29.0
DELTA_3 = DELTA ** 3


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """sRGB gama düzeltmesini kaldırarak doğrusal RGB uzayına çevirir."""
    norm = rgb / 255.0
    mask = norm > 0.04045
    linear = np.empty_like(norm)
    linear[mask] = ((norm[mask] + 0.055) / 1.055) ** 2.4
    linear[~mask] = norm[~mask] / 12.92
    return linear


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Doğrusal RGB'yi sRGB gama uzayına çevirir."""
    mask = linear > 0.0031308
    srgb = np.empty_like(linear)
    srgb[mask] = 1.055 * (linear[mask] ** (1.0 / 2.4)) - 0.055
    srgb[~mask] = 12.92 * linear[~mask]
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def _f_transform(t: np.ndarray) -> np.ndarray:
    mask = t > DELTA_3
    f_t = np.empty_like(t)
    f_t[mask] = np.cbrt(t[mask])
    f_t[~mask] = (t[~mask] / (3.0 * (DELTA ** 2))) + (4.0 / 29.0)
    return f_t


def _f_inv_transform(t: np.ndarray) -> np.ndarray:
    mask = t > DELTA
    inv_t = np.empty_like(t)
    inv_t[mask] = t[mask] ** 3
    inv_t[~mask] = 3.0 * (DELTA ** 2) * (t[~mask] - 4.0 / 29.0)
    return inv_t


def rgb_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    """
    RGB renk dizisini (H, W, 3) veya (N, 3) CIELAB (L*, a*, b*) uzayına dönüştürür.
    L* in [0, 100], a* in [-128, 127], b* in [-128, 127].
    """
    rgb_arr = np.asarray(rgb_array)
    shape = rgb_arr.shape
    flat_rgb = rgb_arr.reshape(-1, 3).astype(np.float64)

    # 1. Doğrusal RGB
    lin = _srgb_to_linear(flat_rgb)

    # 2. sRGB -> XYZ (D65 Dönüşüm Matrisi)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float64)

    xyz = np.dot(lin, M.T)

    # 3. XYZ -> LAB
    x_r = xyz[:, 0] / X_N
    y_r = xyz[:, 1] / Y_N
    z_r = xyz[:, 2] / Z_N

    fx = _f_transform(x_r)
    fy = _f_transform(y_r)
    fz = _f_transform(z_r)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    lab = np.stack([L, a, b], axis=-1)
    return lab.reshape(shape)


def lab_to_rgb(lab_array: np.ndarray) -> np.ndarray:
    """CIELAB (L*, a*, b*) dizisini sRGB (0-255 uint8) dizisine dönüştürür."""
    lab_arr = np.asarray(lab_array)
    shape = lab_arr.shape
    flat_lab = lab_arr.reshape(-1, 3).astype(np.float64)

    L = flat_lab[:, 0]
    a = flat_lab[:, 1]
    b = flat_lab[:, 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    xr = _f_inv_transform(fx)
    yr = _f_inv_transform(fy)
    zr = _f_inv_transform(fz)

    xyz = np.stack([xr * X_N, yr * Y_N, zr * Z_N], axis=-1)

    # XYZ -> Linear RGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=np.float64)

    lin = np.dot(xyz, M_inv.T)
    rgb = _linear_to_srgb(lin)
    return rgb.reshape(shape)
