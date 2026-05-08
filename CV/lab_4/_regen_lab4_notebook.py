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
# Лабораторная работа 4 — Компьютерное зрение

Использованные материалы:
- `docs/Компьютерное зрение лаба 4.pptx.pdf`
- `docs/Компьютерное зрение_5.pptx.pdf`
- `docs/Компьютерное зрение_6.pdf`

В работе реализованы:
1. Склейка панорамы `stitch_images(images, ...)`.
2. Оптический поток Лукаса-Канаде `optic_flow(img1, img2, window_size)`.
"""
    )
)

cells.append(
    code_cell(
        """
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
import imageio_ffmpeg

plt.rcParams["figure.figsize"] = (12, 7)
np.random.seed(42)
plt.rcParams["animation.writer"] = "ffmpeg"
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
"""
    )
)

cells.append(
    md_cell(
        """
## Задание 1. Склейка панорамы

Шаги по заданию:
1. Изменить размер изображений (до 640x640).
2. Перевести в grayscale.
3. Найти SIFT ключевые точки и дескрипторы.
4. Сопоставить дескрипторы.
5. Найти гомографию.
6. Деформировать второе изображение и склеить.
"""
    )
)

cells.append(
    code_cell(
        """
def resize_max_side(image, max_side=640):
    h, w = image.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return image.copy()
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_panorama_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([p for p in Path(folder).glob("*") if p.suffix.lower() in exts])
    images = []
    for p in files:
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(resize_max_side(rgb, max_side=640))
    return files, images


PANORAMA_DIR = Path("panorama")
panorama_files, input_images = load_panorama_images(PANORAMA_DIR)

print(f"Фото для панорамы: {len(input_images)}")
for p in panorama_files:
    print(" -", p.name)

for i, img in enumerate(input_images):
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.title(f"Входное изображение #{i + 1}")
    plt.axis("off")
    plt.show()
"""
    )
)

cells.append(
    code_cell(
        """
def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def detect_and_describe(gray):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(des1, des2, ratio=0.75):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in pairs:
        if m.distance < ratio * n.distance:
            good.append(m)
    good.sort(key=lambda m: m.distance)
    return good


def draw_keypoints_image(image, keypoints):
    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(40, 255, 40),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_matches_image(image1, kp1, image2, kp2, matches, mask=None, max_draw=100):
    matches_to_draw = matches[:max_draw]
    draw_mask = None if mask is None else [int(v) for v in mask[: len(matches_to_draw)]]
    return cv2.drawMatches(
        image1,
        kp1,
        image2,
        kp2,
        matches_to_draw,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=draw_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def blend_images(base_canvas, warped_canvas, base_mask, warped_mask):
    out = np.zeros_like(base_canvas)
    only_base = base_mask & ~warped_mask
    only_warped = warped_mask & ~base_mask
    overlap = base_mask & warped_mask

    out[only_base] = base_canvas[only_base]
    out[only_warped] = warped_canvas[only_warped]
    out[overlap] = (
        0.5 * base_canvas[overlap].astype(np.float32) +
        0.5 * warped_canvas[overlap].astype(np.float32)
    ).astype(np.uint8)
    return out


def crop_nonzero(image):
    mask = np.any(image > 0, axis=2)
    ys, xs = np.where(mask)
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def stitch_pair(image_left, image_right, ratio=0.75, reproj_thresh=4.0, min_matches=20, min_inliers=12):
    gray_left = to_gray(image_left)
    gray_right = to_gray(image_right)

    kp_left, des_left = detect_and_describe(gray_left)
    kp_right, des_right = detect_and_describe(gray_right)
    matches = match_descriptors(des_left, des_right, ratio=ratio)

    debug = {
        "kp_left": kp_left,
        "kp_right": kp_right,
        "matches": matches,
        "inlier_mask": None,
        "num_matches": len(matches),
        "num_inliers": 0,
        "status": "ok",
    }

    if len(matches) < min_matches:
        debug["status"] = "few_matches"
        return None, debug

    pts_left = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, reproj_thresh)

    if H is None or mask is None:
        debug["status"] = "homography_failed"
        return None, debug

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    debug["inlier_mask"] = inlier_mask
    debug["num_inliers"] = inliers

    if inliers < min_inliers:
        debug["status"] = "few_inliers"
        return None, debug

    h_l, w_l = image_left.shape[:2]
    h_r, w_r = image_right.shape[:2]

    corners_left = np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]]).reshape(-1, 1, 2)
    corners_right = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
    warped_right_corners = cv2.perspectiveTransform(corners_right, H)
    all_corners = np.concatenate([corners_left, warped_right_corners], axis=0)

    xy_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    xy_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    xmin, ymin = xy_min.tolist()
    xmax, ymax = xy_max.tolist()

    tx, ty = -xmin, -ymin
    T = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)

    out_w = int(xmax - xmin)
    out_h = int(ymax - ymin)

    warped_right = cv2.warpPerspective(image_right, T @ H, (out_w, out_h))
    warped_mask = cv2.warpPerspective(np.ones((h_r, w_r), dtype=np.uint8), T @ H, (out_w, out_h)) > 0

    left_canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    left_mask = np.zeros((out_h, out_w), dtype=bool)
    left_canvas[ty:ty + h_l, tx:tx + w_l] = image_left
    left_mask[ty:ty + h_l, tx:tx + w_l] = True

    stitched = blend_images(left_canvas, warped_right, left_mask, warped_mask)
    stitched = crop_nonzero(stitched)
    return stitched, debug


def stitch_images(images, ratio=0.75, reproj_thresh=4.0, min_matches=20, min_inliers=12):
    current = images[0]
    debug_info = []
    for i in range(1, len(images)):
        stitched, dbg = stitch_pair(
            current,
            images[i],
            ratio=ratio,
            reproj_thresh=reproj_thresh,
            min_matches=min_matches,
            min_inliers=min_inliers,
        )
        debug_info.append(dbg)
        if stitched is None:
            return None, debug_info
        current = stitched
    return current, debug_info
"""
    )
)

cells.append(
    code_cell(
        """
def show_image(image, title, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


# Ключевые точки на каждом изображении
all_kp = []
all_des = []
for img in input_images:
    kp, des = detect_and_describe(to_gray(img))
    all_kp.append(kp)
    all_des.append(des)
    kp_img = draw_keypoints_image(img, kp)
    show_image(kp_img, f"Ключевые точки #{len(all_kp)}: {len(kp)}")


# Сопоставления между соседними изображениями
for i in range(len(input_images) - 1):
    matches = match_descriptors(all_des[i], all_des[i + 1], ratio=0.75)
    match_img = draw_matches_image(
        input_images[i],
        all_kp[i],
        input_images[i + 1],
        all_kp[i + 1],
        matches,
        None,
        max_draw=100,
    )
    plt.figure(figsize=(18, 6))
    plt.imshow(match_img)
    plt.title(f"Сопоставления #{i + 1} -> #{i + 2}: {len(matches)}")
    plt.axis("off")
    plt.show()


# Склейка панорамы
panorama, pano_debug = stitch_images(
    input_images,
    ratio=0.75,
    reproj_thresh=4.0,
    min_matches=20,
    min_inliers=12,
)

if panorama is None:
    print("Склейка не выполнена: пересечения недостаточно.")
else:
    print("Склейка выполнена.")
    for i, dbg in enumerate(pano_debug, start=1):
        print(f"Пара {i}: matches={dbg['num_matches']}, inliers={dbg['num_inliers']}, status={dbg['status']}")
    for i, img in enumerate(input_images):
        show_image(img, f"Фото #{i + 1}")
    show_image(panorama, "Панорама")
"""
    )
)

cells.append(
    md_cell(
        """
## Задание 2. Оптический поток Лукаса-Канаде

Шаги по заданию:
1. Загрузить видео и перевести кадры в grayscale.
2. Реализовать `optic_flow(img1, img2, window_size)`.
3. Показать результат по кадрам.
"""
    )
)

cells.append(
    code_cell(
        """
def display_video(images, interval=90, cmap=None):
    fig = plt.figure(figsize=(8, 5))
    plt.axis("off")
    if images[0].ndim == 2:
        im = plt.imshow(images[0], cmap="gray" if cmap is None else cmap)
    else:
        im = plt.imshow(images[0])
    plt.close()

    def init():
        im.set_data(images[0])
        return (im,)

    def animate(i):
        im.set_data(images[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(images),
        interval=interval,
        blit=True,
    )
    display(HTML(anim.to_html5_video()))
"""
    )
)

cells.append(
    code_cell(
        """
def load_video_frames(video_path, max_frames=48, sample_every=3, max_side=480):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = resize_max_side(frame_rgb, max_side=max_side)
            frames.append(frame_rgb)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


VIDEO_PATH = Path("пудж флексит (pudge dance) (1080p).mp4")
frames_rgb = load_video_frames(VIDEO_PATH, max_frames=48, sample_every=3, max_side=480)
frames_gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for f in frames_rgb]

print(f"Кадров для анализа: {len(frames_gray)}")
display_video(frames_rgb[: min(24, len(frames_rgb))], interval=90)
"""
    )
)

cells.append(
    code_cell(
        """
def optic_flow(img1, img2, window_size=11):
    Ix = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    It = img2 - img1

    k = (window_size, window_size)
    Sxx = cv2.boxFilter(Ix * Ix, cv2.CV_32F, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    Syy = cv2.boxFilter(Iy * Iy, cv2.CV_32F, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    Sxy = cv2.boxFilter(Ix * Iy, cv2.CV_32F, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    Sxt = cv2.boxFilter(Ix * It, cv2.CV_32F, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    Syt = cv2.boxFilter(Iy * It, cv2.CV_32F, k, normalize=False, borderType=cv2.BORDER_REFLECT)

    det = Sxx * Syy - Sxy * Sxy
    eps = 1e-4
    valid = det > eps

    u = np.zeros_like(img1, dtype=np.float32)
    v = np.zeros_like(img1, dtype=np.float32)

    u[valid] = (-Syy[valid] * Sxt[valid] + Sxy[valid] * Syt[valid]) / det[valid]
    v[valid] = (Sxy[valid] * Sxt[valid] - Sxx[valid] * Syt[valid]) / det[valid]

    return np.stack([u, v], axis=-1)


def flow_to_rgb(flow, clip_percentile=98):
    u = flow[..., 0]
    v = flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=False)
    vmax = np.percentile(mag, clip_percentile)
    vmax = max(vmax, 1e-6)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang / (2 * np.pi)) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip((mag / vmax) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def draw_flow_arrows(gray_frame, flow, step=14, scale=2.5, min_mag=0.08):
    base = np.clip(gray_frame * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    h, w = gray_frame.shape
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            u, v = flow[y, x]
            if np.hypot(u, v) < min_mag:
                continue
            x2 = int(round(x + u * scale))
            y2 = int(round(y + v * scale))
            cv2.arrowedLine(vis, (x, y), (x2, y2), (255, 80, 80), 1, tipLength=0.3, line_type=cv2.LINE_AA)
    return vis
"""
    )
)

cells.append(
    code_cell(
        """
flows = []
flow_rgb_frames = []
flow_arrow_frames = []

for i in range(len(frames_gray) - 1):
    flow = optic_flow(frames_gray[i], frames_gray[i + 1], window_size=11)
    flows.append(flow)
    flow_rgb_frames.append(flow_to_rgb(flow))
    flow_arrow_frames.append(draw_flow_arrows(frames_gray[i], flow))

print(f"Карт потока построено: {len(flows)}")
print(f"Форма карты потока: {flows[0].shape} (H, W, 2)")

display_video(flow_rgb_frames[: min(24, len(flow_rgb_frames))], interval=90)
display_video(flow_arrow_frames[: min(24, len(flow_arrow_frames))], interval=90)

u0 = flows[0][..., 0]
v0 = flows[0][..., 1]
mag0 = np.sqrt(u0 * u0 + v0 * v0)

plt.figure(figsize=(12, 6))
plt.imshow(u0, cmap="coolwarm")
plt.title("u (смещение по X), кадр 0->1")
plt.axis("off")
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(v0, cmap="coolwarm")
plt.title("v (смещение по Y), кадр 0->1")
plt.axis("off")
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(mag0, cmap="inferno")
plt.title("|V| (модуль потока), кадр 0->1")
plt.axis("off")
plt.show()
"""
    )
)

cells.append(
    md_cell(
        """
## Вывод

- Реализована склейка панорамы `stitch_images(images, ...)` по алгоритму из задания.
- Показаны входные изображения, ключевые точки и сопоставления между изображениями.
- Реализован оптический поток Лукаса-Канаде `optic_flow(img1, img2, window_size)`.
- Показаны карты смещений по кадрам для последовательности из видео (использовано больше 5 кадров).
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

out = Path("CV/lab_4/lab_4.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Создан: {out}")
