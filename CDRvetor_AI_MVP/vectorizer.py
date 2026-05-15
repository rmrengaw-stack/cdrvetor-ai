import cv2
import numpy as np
from pathlib import Path


def _hex_from_bgr(color):
    b, g, r = [int(v) for v in color]
    return f"#{r:02x}{g:02x}{b:02x}"


def _polygon_path(points):
    if len(points) < 3:
        return ""

    d = f"M {points[0][0]:.2f} {points[0][1]:.2f} "
    for x, y in points[1:]:
        d += f"L {x:.2f} {y:.2f} "
    d += "Z"
    return d


def _bezier_path(points, smoothness=0.12):
    if len(points) < 4:
        return _polygon_path(points)

    d = f"M {points[0][0]:.2f} {points[0][1]:.2f} "
    n = len(points)

    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i % n]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]

        c1 = p1 + (p2 - p0) * smoothness
        c2 = p2 - (p3 - p1) * smoothness

        d += (
            f"C {c1[0]:.2f} {c1[1]:.2f}, "
            f"{c2[0]:.2f} {c2[1]:.2f}, "
            f"{p2[0]:.2f} {p2[1]:.2f} "
        )

    d += "Z"
    return d


def _contour_to_path(contour, area, perimeter, quality="balanceado"):
    """
    V5: qualidade ajustável.
    - economico: menos pontos, SVG mais leve
    - balanceado: equilíbrio
    - ultra: preserva mais detalhes
    """
    if quality == "economico":
        small_eps, mid_eps, big_eps = 0.0014, 0.0020, 0.0030
        smooth_small, smooth_mid, smooth_big = 0.06, 0.10, 0.16
    elif quality == "ultra":
        small_eps, mid_eps, big_eps = 0.00055, 0.0010, 0.0016
        smooth_small, smooth_mid, smooth_big = 0.04, 0.08, 0.12
    else:
        small_eps, mid_eps, big_eps = 0.00085, 0.00135, 0.0021
        smooth_small, smooth_mid, smooth_big = 0.05, 0.09, 0.14

    if area < 2200:
        epsilon_factor = small_eps
        smoothness = smooth_small
    elif area < 10000:
        epsilon_factor = mid_eps
        smoothness = smooth_mid
    else:
        epsilon_factor = big_eps
        smoothness = smooth_big

    epsilon = max(0.30, epsilon_factor * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2).astype(float)

    if len(points) < 3:
        return ""

    # Textos e detalhes pequenos ficam melhores com linha, não Bézier excessiva.
    if area < 2600:
        return _polygon_path(points)

    return _bezier_path(points, smoothness=smoothness)


def _color_distance(c1, c2):
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    return float(np.linalg.norm(c1 - c2))


def _detect_background_color(image):
    h, w = image.shape[:2]
    border = np.concatenate([
        image[0:10, :, :].reshape(-1, 3),
        image[h-10:h, :, :].reshape(-1, 3),
        image[:, 0:10, :].reshape(-1, 3),
        image[:, w-10:w, :].reshape(-1, 3),
    ], axis=0)
    return np.median(border, axis=0).astype(np.uint8)


def _is_near_white(color, threshold=244):
    b, g, r = [int(v) for v in color]
    return r > threshold and g > threshold and b > threshold


def _protect_dark_details(image, labels, centers):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, dark = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Mantém detalhes escuros, mas não engorda demais.
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    dark_pixels = image[dark > 0]
    if len(dark_pixels) < 20:
        return labels, centers

    dark_color = np.median(dark_pixels, axis=0).astype(np.uint8)

    counts = [int(np.sum(labels == i)) for i in range(len(centers))]
    replace_index = int(np.argmin(counts))
    centers[replace_index] = dark_color
    labels[dark > 0] = replace_index

    return labels, centers


def _clean_mask(mask, min_area, preserve_details=False, quality="balanceado"):
    kernel = np.ones((2, 2), np.uint8)

    if preserve_details or quality == "ultra":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Edge-aware: recupera bordas importantes sem engordar muito.
    edges = cv2.Canny(mask, 45, 135)
    if quality == "ultra":
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    mask = cv2.bitwise_or(mask, edges)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)

    return clean


def _add_paths_from_mask(svg_parts, mask, fill, min_area, preserve_details=False, quality="balanceado"):
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or not contours:
        return

    hierarchy = hierarchy[0]

    svg_parts.append(
        f'<g id="camada_cor_{fill.replace("#", "")}" fill="{fill}" stroke="none" fill-rule="evenodd">'
    )

    used = set()

    for idx, contour in sorted(
        enumerate(contours),
        key=lambda item: cv2.contourArea(item[1]),
        reverse=True
    ):
        if idx in used:
            continue

        parent = hierarchy[idx][3]
        if parent != -1:
            continue

        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        d = _contour_to_path(contour, area, perimeter, quality=quality)

        if not d:
            continue

        used.add(idx)

        child = hierarchy[idx][2]
        while child != -1:
            hole = contours[child]
            hole_area = cv2.contourArea(hole)

            # Ultra preserva furos menores.
            if quality == "ultra":
                hole_limit = max(2, min_area * 0.12)
            else:
                hole_limit = max(3, min_area * (0.18 if preserve_details else 0.30))

            if hole_area >= hole_limit:
                hole_perimeter = cv2.arcLength(hole, True)
                hole_d = _contour_to_path(hole, hole_area, hole_perimeter, quality=quality)
                if hole_d:
                    d += " " + hole_d

            used.add(child)
            child = hierarchy[child][0]

        svg_parts.append(f'<path d="{d}"/>')

    svg_parts.append("</g>")


def vectorize_image(input_path: str, output_path: str, quality: str = "balanceado", transparent: bool = False):
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Não foi possível abrir a imagem.")

    quality = (quality or "balanceado").lower()
    if quality not in ["economico", "balanceado", "ultra"]:
        quality = "balanceado"

    height, width = image.shape[:2]
    total_area = width * height

    image = cv2.bilateralFilter(image, 7, 42, 42)

    blur = cv2.GaussianBlur(image, (0, 0), 0.9)
    image = cv2.addWeighted(image, 1.18, blur, -0.18, 0)

    background = _detect_background_color(image)

    k = 10 if quality == "economico" else 14 if quality == "balanceado" else 18
    pixels = image.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        45 if quality != "ultra" else 60,
        0.65
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    labels = labels.flatten().reshape((height, width))

    labels, centers = _protect_dark_details(image, labels, centers)

    min_area = max(
        4 if quality == "ultra" else 6,
        int(total_area * (0.000012 if quality == "ultra" else 0.000018))
    )

    bg_fill = _hex_from_bgr(background)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<metadata>CDRvetor AI V5 - quality={quality}, transparent={transparent}</metadata>'
    ]

    if not transparent:
        svg_parts.append(f'<rect id="background" width="100%" height="100%" fill="{bg_fill}"/>')

    color_order = []
    for i, color in enumerate(centers):
        count = int(np.sum(labels == i))
        color_order.append((count, i, color))

    color_order.sort(reverse=True)

    for _, color_index, color in color_order:
        # Fundo transparente ou fundo sólido: em ambos os casos não vetoriza fundo.
        if _is_near_white(color) or _color_distance(color, background) < 18:
            continue

        fill = _hex_from_bgr(color)
        preserve_details = np.mean(color) < 125

        mask = (labels == color_index).astype(np.uint8) * 255
        mask = _clean_mask(mask, min_area, preserve_details=preserve_details, quality=quality)

        _add_paths_from_mask(
            svg_parts,
            mask,
            fill,
            min_area,
            preserve_details=preserve_details,
            quality=quality
        )

    svg_parts.append("</svg>")

    Path(output_path).write_text("\n".join(svg_parts), encoding="utf-8")
