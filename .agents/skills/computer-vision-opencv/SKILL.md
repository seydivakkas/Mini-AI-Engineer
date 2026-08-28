---
name: computer-vision-opencv
description: "Master classical computer vision and OpenCV pipelines: image filtering, color spaces (RGB, HSV, LAB, CIELAB, Delta-E), geometric transformations, morphological operations, contour detection, feature extraction (SIFT, ORB), edge detection, camera calibration, optical flow, video processing, and hybrid CV+DL architectures."
risk: unknown
source: community
date_added: '2026-02-28'
---

# Computer Vision & OpenCV Masterclass

Expert guidance and production-grade architectures for classical computer vision, industrial image processing, colorimetry, frequency domain analysis, and real-time vision pipelines using OpenCV and NumPy.

## When to Use This Skill

Use this skill when:
- Designing low-latency ($<10\text{ ms}$) industrial inspection, automated optical inspection (AOI), or defect detection systems
- Performing color space conversions, perceptual color tolerance calculations (CIELAB, $\Delta E_{00}$, HSV masking)
- Applying spatial filtering (Bilateral, Gaussian, Median) or morphological operations (Top-Hat, Black-Hat, Opening, Closing)
- Detecting edges, contours, blobs, convex hulls, and geometric bounding boxes
- Performing feature detection, description, and matching (SIFT, ORB, FAST, FLANN, BFMatcher, RANSAC Homography)
- Conducting frequency domain analysis (2D FFT, high-frequency power ratios, Laplacian variance blur/focus detection)
- Handling camera calibration, lens distortion correction, perspective warps, and geometric remapping
- Tracking objects, computing dense/sparse optical flow (Lucas-Kanade, Farnebäck), or background subtraction (MOG2, KNN)
- Preprocessing and augmenting images before feeding them into deep learning architectures (YOLO, ViT, CNNs)

---

## Core Capabilities & Technical Pillars

### 1. Color Spaces & Perceptual Colorimetry

OpenCV reads images in **BGR format** by default. Converting between color spaces is critical for lighting-invariant segmentation and color tolerance.

```python
import cv2
import numpy as np

# BGR to RGB / Gray / HSV / LAB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
```

- **HSV / HLS (Hue, Saturation, Value/Lightness):** Isolates color chroma (Hue) from lighting intensity (Value). Ideal for color thresholding and object tracking under variable shadows.
- **CIELAB ($L^*, a^*, b^*$):** Perceptually uniform color space where Euclidean distances approximate human visual difference.
  - $L^* \in [0, 100]$: Perceived Lightness.
  - $a^* \in [-128, +127]$: Green-to-Red axis.
  - $b^* \in [-128, +127]$: Blue-to-Yellow axis.
- **Color Difference Metrics ($\Delta E$):**
  - $\Delta E_{76} = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}$ (Euclidean approximation)
  - $\Delta E_{2000}$ ($\text{CIEDE2000}$): Standard industrial metric accounting for lightness, chroma, and hue non-uniformities plus blue-region rotation.

---

### 2. Spatial Filtering & Morphological Operations

#### Edge-Preserving and Noise Reduction Filters
```python
# Gaussian Blur (linear smoothing)
blurred = cv2.GaussianBlur(img_gray, (5, 5), sigmaX=1.5)

# Median Blur (salt-and-pepper noise removal)
median = cv2.medianBlur(img_gray, 5)

# Bilateral Filter (smooths textures while strictly preserving sharp edges)
bilateral = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)
```

#### Morphological Transformations
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Top-Hat: Isolates bright elements/scratches on dark or textured backgrounds
tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)

# Black-Hat: Isolates dark elements/stains/holes on bright backgrounds
blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

# Opening (Erosion followed by Dilation): Removes isolated small white noise pixels
opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# Closing (Dilation followed by Erosion): Closes small holes and bridges gaps
closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
```

---

### 3. Edge, Contour & Shape Analysis

```python
# Canny Edge Detector with automatic or adaptive Otsu thresholds
edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)

# Finding Contours with Hierarchical Trees
contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 30:
        continue
    
    # Bounding Rect & Min-Area Rotated Rect
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)  # (center, (w, h), angle)
    
    # Polygon Approximation & Convex Hull
    perimeter = cv2.arcLength(cnt, closed=True)
    approx = cv2.approxPolyDP(cnt, epsilon=0.02 * perimeter, closed=True)
    hull = cv2.convexHull(cnt)
    
    # Circularity & Aspect Ratio
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    aspect_ratio = float(w) / max(h, 1)
```

---

### 4. 2D Frequency Domain & Blur / Focus Detection

#### A. Laplacian Variance Focus Measure
$$\text{Focus Measure} = \text{Var}(\nabla^2 I)$$
```python
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
focus_variance = float(laplacian.var())
is_sharp = focus_variance > 100.0
```

#### B. 2D Fast Fourier Transform (FFT) Power Spectrum
```python
f_transform = np.fft.fft2(img_gray.astype(float))
f_shift = np.fft.fftshift(f_transform)
magnitude = np.abs(f_shift)

# High-frequency power ratio (masks out low-frequency center)
h, w = img_gray.shape
cy, cx = h // 2, w // 2
radius = min(h, w) // 10
y, x = np.ogrid[:h, :w]
mask_low = ((x - cx)**2 + (y - cy)**2) <= radius**2

total_energy = np.sum(magnitude**2) + 1e-8
low_energy = np.sum((magnitude * mask_low)**2)
high_freq_ratio = (total_energy - low_energy) / total_energy * 100.0
```

---

### 5. Feature Detection, Descriptors & Matching

```python
# ORB (Fast, rotation-invariant, patent-free alternative to SIFT)
orb = cv2.ORB_create(nfeatures=1000)
keypoints, descriptors = orb.detectAndCompute(img_gray, None)

# FLANN or BFMatcher with Lowe's Ratio Test
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(desc1, desc2, k=2)

good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Find Homography Matrix with RANSAC
if len(good_matches) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

---

### 6. Geometric Transformations & Perspective Correction

```python
# 4-Point Perspective Transform (Document Scanner / Top-Down View)
src_corners = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
dst_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

M_perspective = cv2.getPerspectiveTransform(src_corners, dst_corners)
warped = cv2.warpPerspective(img_rgb, M_perspective, (width, height))
```

---

## Best Practices & Engineering Rules

1. **Avoid Python Loops Over Pixels:** Always vectorize operations using NumPy slice notation or native OpenCV C++ functions (`cv2.inRange`, `cv2.addWeighted`, `cv2.LUT`, `cv2.bitwise_and`).
2. **Handle BGR vs RGB Explicitly:** When passing OpenCV images to Matplotlib, Pillow, or PyTorch, always convert via `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)`.
3. **Prevent In-Place Mutation Bugs:** Functions modifying images (e.g., drawing bounding boxes or contours) must operate on explicit copies (`img.copy()`) to prevent unexpected side effects.
4. **Memory Contiguity:** After cropping, flipping, or slicing arrays, ensure memory is C-contiguous using `np.ascontiguousarray()` before passing to OpenCV functions.
5. **Robust Thresholding:** Use Otsu's thresholding (`cv2.THRESH_OTSU`) or Adaptive Gaussian Thresholding (`cv2.ADAPTIVE_THRESH_GAUSSIAN_C`) when dealing with non-uniform ambient illumination.
