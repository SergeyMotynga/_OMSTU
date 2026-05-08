import json
from pathlib import Path


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


cells = []

cells.append(
    md_cell(
        """
# Лабораторная работа 3 — Компьютерное зрение

Использованные материалы:
- `Computer_Vision_lab3.pptx.pdf`
- `Компьютерное_зрение_4_лекция.pptx.pdf`

В работе реализованы:
1. Генерация шумов и фильтрация изображения.
2. Детекция границ методом Canny (пошагово по лекции).
3. Детектор углов Харриса `detect_harris(img, ...)`.
"""
    )
)

cells.append(
    code_cell(
        """
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

plt.rcParams["figure.figsize"] = (12, 7)
np.random.seed(42)
"""
    )
)

cells.append(
    code_cell(
        """
img_bgr = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 7))
plt.imshow(img_rgb)
plt.title("Исходное изображение (RGB)")
plt.axis("off")
plt.show()

plt.figure(figsize=(12, 7))
plt.imshow(img_gray, cmap="gray")
plt.title("Серое изображение")
plt.axis("off")
plt.show()
"""
    )
)

cells.append(
    md_cell(
        """
## Задание 1. Работа с шумом

Требуется:
- Сгенерировать шумы `salt`, `pepper`, `gaussian` с разными параметрами.
- Наложить шум на изображение и отобразить результаты.
- Применить фильтры: Гаусса, Box filter, медианный.
- Для фильтра Гаусса использовать несколько параметров.
"""
    )
)

cells.append(
    code_cell(
        """
def show_images(images, titles, figsize=(12, 7), cmap=None):
    for image, title in zip(images, titles):
        plt.figure(figsize=figsize)
        if cmap is None:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap=cmap)
        plt.title(title, fontsize=12)
        plt.axis("off")
        plt.show()


def add_salt_noise(image, amount=0.01, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    noisy = image.copy()
    h, w = noisy.shape[:2]
    num = int(amount * h * w)
    ys = rng.integers(0, h, size=num)
    xs = rng.integers(0, w, size=num)
    noisy[ys, xs, :] = 255
    return noisy


def add_pepper_noise(image, amount=0.01, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    noisy = image.copy()
    h, w = noisy.shape[:2]
    num = int(amount * h * w)
    ys = rng.integers(0, h, size=num)
    xs = rng.integers(0, w, size=num)
    noisy[ys, xs, :] = 0
    return noisy


def add_salt_pepper_noise(image, salt_amount=0.01, pepper_amount=0.01, rng=None):
    noisy = add_salt_noise(image, amount=salt_amount, rng=rng)
    noisy = add_pepper_noise(noisy, amount=pepper_amount, rng=rng)
    return noisy


def add_gaussian_noise(image, sigma=10.0, mean=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(loc=mean, scale=sigma, size=image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_blur_uint8(image, sigma):
    k = int(2 * round(3 * sigma) + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)
"""
    )
)

cells.append(
    code_cell(
        """
rng = np.random.default_rng(42)

salt_levels = [0.005, 0.015, 0.03]
pepper_levels = [0.005, 0.015, 0.03]
gaussian_sigmas = [10, 25, 40]

salt_images = [add_salt_noise(img_rgb, amount=a, rng=rng) for a in salt_levels]
pepper_images = [add_pepper_noise(img_rgb, amount=a, rng=rng) for a in pepper_levels]
gauss_images = [add_gaussian_noise(img_rgb, sigma=s, rng=rng) for s in gaussian_sigmas]

show_images(
    [img_rgb] + salt_images,
    ["Оригинал"] + [f"Шум соль, amount={a}" for a in salt_levels],
)

show_images(
    [img_rgb] + pepper_images,
    ["Оригинал"] + [f"Шум перец, amount={a}" for a in pepper_levels],
)

show_images(
    [img_rgb] + gauss_images,
    ["Оригинал"] + [f"Гауссов шум, sigma={s}" for s in gaussian_sigmas],
)
"""
    )
)

cells.append(
    code_cell(
        """
sp_noisy = add_salt_pepper_noise(
    img_rgb,
    salt_amount=0.02,
    pepper_amount=0.02,
    rng=np.random.default_rng(7),
)
g_noisy = add_gaussian_noise(img_rgb, sigma=25, rng=np.random.default_rng(7))


def apply_denoise_set(noisy_image):
    gauss_sigmas = [0.8, 1.5, 2.5]
    results = [("Шумное изображение", noisy_image)]
    for s in gauss_sigmas:
        results.append((f"Фильтр Гаусса, sigma={s}", gaussian_blur_uint8(noisy_image, sigma=s)))
    results.append(("Box filter 5x5", cv2.blur(noisy_image, (5, 5))))
    results.append(("Медианный фильтр 5x5", cv2.medianBlur(noisy_image, 5)))
    return results


sp_results = apply_denoise_set(sp_noisy)
g_results = apply_denoise_set(g_noisy)

show_images([img for _, img in sp_results], [title for title, _ in sp_results])
show_images([img for _, img in g_results], [title for title, _ in g_results])
"""
    )
)

cells.append(
    md_cell(
        """
## Задание 2. Детекция границ (Canny)

Реализация по шагам из лекции:
1. Сглаживание изображения.
2. Производные по `x` и `y`, магнитуда и направление градиента.
3. Non-maximum suppression.
4. Двойной порог: сильные и слабые границы.
5. Связывание границ (hysteresis).
"""
    )
)

cells.append(
    code_cell(
        """
def gaussian_blur_float(gray, sigma=1.2):
    k = int(2 * round(3 * sigma) + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(gray, (k, k), sigmaX=sigma, sigmaY=sigma)


def compute_gradients(gray_float):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = cv2.filter2D(gray_float, cv2.CV_32F, sobel_x)
    gy = cv2.filter2D(gray_float, cv2.CV_32F, sobel_y)
    mag = np.hypot(gx, gy)
    if mag.max() > 0:
        mag = mag / mag.max()
    angle = np.rad2deg(np.arctan2(gy, gx))
    angle[angle < 0] += 180
    return gx, gy, mag, angle


def non_maximum_suppression(mag, angle):
    h, w = mag.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 0.0
            r = 0.0
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif 22.5 <= a < 67.5:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            elif 112.5 <= a < 157.5:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]
            if mag[i, j] >= q and mag[i, j] >= r:
                out[i, j] = mag[i, j]
    return out


def double_threshold(nms, low_ratio=0.08, high_ratio=0.18, weak=0.5, strong=1.0):
    high = nms.max() * high_ratio
    low = nms.max() * low_ratio
    out = np.zeros_like(nms, dtype=np.float32)
    strong_mask = nms >= high
    weak_mask = (nms >= low) & (nms < high)
    out[strong_mask] = strong
    out[weak_mask] = weak
    return out, weak, strong


def edge_tracking_by_hysteresis(thresholded, weak=0.5, strong=1.0):
    strong_mask = thresholded == strong
    weak_mask = thresholded == weak
    h, w = thresholded.shape
    q = deque(map(tuple, np.argwhere(strong_mask)))
    while q:
        i, j = q.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and weak_mask[ni, nj]:
                    weak_mask[ni, nj] = False
                    strong_mask[ni, nj] = True
                    q.append((ni, nj))
    return strong_mask.astype(np.uint8) * 255


def canny_custom(gray_uint8, sigma=1.4, low_ratio=0.08, high_ratio=0.18):
    gray = gray_uint8.astype(np.float32) / 255.0
    smoothed = gaussian_blur_float(gray, sigma=sigma)
    gx, gy, mag, angle = compute_gradients(smoothed)
    nms = non_maximum_suppression(mag, angle)
    thresholded, weak, strong = double_threshold(
        nms, low_ratio=low_ratio, high_ratio=high_ratio
    )
    edges = edge_tracking_by_hysteresis(thresholded, weak=weak, strong=strong)
    return {
        "smoothed": smoothed,
        "gx": gx,
        "gy": gy,
        "mag": mag,
        "nms": nms,
        "thresholded": thresholded,
        "edges": edges,
    }
"""
    )
)

cells.append(
    code_cell(
        """
canny_res = canny_custom(img_gray, sigma=1.4, low_ratio=0.08, high_ratio=0.18)

plots = [
    (img_gray, "Серое изображение", "gray"),
    (canny_res["gx"], "Производная по X (Ix)", "seismic"),
    (canny_res["gy"], "Производная по Y (Iy)", "seismic"),
    (canny_res["mag"], "Магнитуда градиента", "gray"),
    (canny_res["nms"], "После подавления немаксимумов", "gray"),
    (canny_res["edges"], "Итог детектора Canny", "gray"),
]

for image, title, cmap in plots:
    plt.figure(figsize=(12, 7))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()
"""
    )
)

cells.append(
    md_cell(
        """
## Задание 3. Детектор углов Харриса

Реализация на сером изображении со сглаживанием и формулой из задания:

`R = det(S) - α * trace(S)^2`, где `α = 0.04...0.06`.
"""
    )
)

cells.append(
    code_cell(
        """
def detect_harris(gray_uint8, k=0.05, sigma=1.5, window_size=5, threshold_rel=0.02, nms_window=5):
    if window_size % 2 == 0:
        window_size += 1
    if nms_window % 2 == 0:
        nms_window += 1

    gray = gray_uint8.astype(np.float32) / 255.0
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)

    ix = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

    ixx = ix * ix
    iyy = iy * iy
    ixy = ix * iy

    sxx = cv2.GaussianBlur(ixx, (window_size, window_size), sigmaX=sigma, sigmaY=sigma)
    syy = cv2.GaussianBlur(iyy, (window_size, window_size), sigmaX=sigma, sigmaY=sigma)
    sxy = cv2.GaussianBlur(ixy, (window_size, window_size), sigmaX=sigma, sigmaY=sigma)

    det_s = sxx * syy - sxy * sxy
    trace_s = sxx + syy
    r = det_s - k * (trace_s ** 2)

    threshold = threshold_rel * r.max()
    corners = r > threshold

    kernel = np.ones((nms_window, nms_window), dtype=np.uint8)
    local_max = cv2.dilate(r, kernel)
    corners = corners & (r == local_max)

    ys, xs = np.where(corners)
    return {"R": r, "smoothed": smoothed, "Ix": ix, "Iy": iy, "ys": ys, "xs": xs}


def draw_corners_on_image(image_rgb, xs, ys, color=(255, 0, 0), radius=2):
    out = image_rgb.copy()
    for x, y in zip(xs, ys):
        cv2.circle(out, (int(x), int(y)), radius, color, 1, lineType=cv2.LINE_AA)
    return out
"""
    )
)

cells.append(
    code_cell(
        """
harris_res = detect_harris(
    img_gray,
    k=0.05,
    sigma=1.5,
    window_size=5,
    threshold_rel=0.02,
    nms_window=5,
)

xs, ys = harris_res["xs"], harris_res["ys"]
print("Количество найденных углов:", len(xs))

r = harris_res["R"]
r_vis = (r - r.min()) / (r.max() - r.min() + 1e-12)
overlay = draw_corners_on_image(img_rgb, xs, ys, color=(255, 0, 0), radius=2)

plots = [
    (img_gray, "Серое изображение", "gray"),
    (r_vis, "Отклик Харриса R (нормированный)", "inferno"),
    (overlay, "Найденные углы Харриса", None),
]

for image, title, cmap in plots:
    plt.figure(figsize=(12, 7))
    if cmap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()
"""
    )
)

cells.append(
    md_cell(
        """
## Вывод

- Сгенерированы и визуализированы шумы `salt`, `pepper`, `gaussian` с несколькими параметрами.
- Применены фильтры Гаусса (с разными `sigma`), Box filter и медианный фильтр.
- Реализован пошаговый Canny: `Ix`, `Iy`, магнитуда, подавление немаксимумов, двойной порог, hysteresis.
- Реализован `detect_harris(img, ...)` и показаны найденные углы на изображении.
"""
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("CV/lab_3/lab_3.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Создан: {out}")
