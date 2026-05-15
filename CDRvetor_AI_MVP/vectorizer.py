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
    return d + "Z"


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

        d += f"C {c1[0]:.2f} {c1[1]:.2f}, {c2[0]:.2f} {c2[1]:.2f}, {p2[0]:.2f} {p2[1]:.2f} "

    return d + "Z"


def _contour_to_path(contour, area, perimeter, quality="balanceado", force_polygon=False):
    if quality == "ultra":
        small_eps, mid_eps, big_eps = 0.00045, 0.0009, 0.0015
    elif quality == "economico":
        small_eps, mid_eps, big_eps = 0.0014, 0.0020, 0.0030
    else:
        small_eps, mid_eps, big_eps = 0.00075, 0.0012, 0.0020

    if area < 2200:
        eps = small_eps
        smooth = 0.04
    elif area < 10000:
        eps = mid_eps
        smooth = 0.08
    else:
        eps = big_eps
        smooth = 0.12

    approx = cv2.approxPolyDP(contour, max(0.25, eps * perimeter), True)
    points = approx.reshape(-1, 2).astype(float)

    if len(points) < 3:
        return ""

    if force_polygon or area < 2600:
        return _polygon_path(points)

    return _bezier_path(points, smoothness=smooth)


def _color_distance(c1, c2):
    return float(np.linalg.norm(np.array(c1, dtype=np.float32) - np.array(c2, dtype=np.float32)))


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


def _dark_detail_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # V6: texto/traços escuros separados antes da vetorização por cor.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Limpa ruído sem fechar furos internos.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def _dominant_color_from_mask(image, mask):
    pixels = image[mask > 0]
    if len(pixels) == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    return np.median(pixels, axis=0).astype(np.uint8)


def _add_paths_from_mask(svg_parts, mask, fill, min_area, quality="balanceado", force_polygon=False, group_name="camada"):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None or not contours:
        return

    hierarchy = hierarchy[0]
    svg_parts.append(f'<g id="{group_name}_{fill.replace("#", "")}" fill="{fill}" stroke="none" fill-rule="evenodd">')

    used = set()

    for idx, contour in sorted(enumerate(contours), key=lambda item: cv2.contourArea(item[1]), reverse=True):
        if idx in used:
            continue

        if hierarchy[idx][3] != -1:
            continue

        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        d = _contour_to_path(contour, area, perimeter, quality=quality, force_polygon=force_polygon)

        if not d:
            continue

        used.add(idx)

        child = hierarchy[idx][2]
        while child != -1:
            hole = contours[child]
            hole_area = cv2.contourArea(hole)

            # Furos pequenos são importantes para texto/números.
            if hole_area >= max(2, min_area * 0.08):
                hole_perimeter = cv2.arcLength(hole, True)
                hole_d = _contour_to_path(hole, hole_area, hole_perimeter, quality=quality, force_polygon=True)
                if hole_d:
                    d += " " + hole_d

            used.add(child)
            child = hierarchy[child][0]

        svg_parts.append(f'<path d="{d}"/>')

    svg_parts.append("</g>")


def _clean_color_mask(mask, min_area, quality="balanceado"):
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if quality == "economico":
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)

    return clean


def vectorize_image(input_path: str, output_path: str, quality: str = "balanceado", transparent: bool = False):
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Não foi possível abrir a imagem.")

    quality = (quality or "balanceado").lower()
    if quality not in ["economico", "balanceado", "ultra"]:
        quality = "balanceado"

    height, width = image.shape[:2]
    total_area = width * height

    image = cv2.bilateralFilter(image, 7, 38, 38)
    blur = cv2.GaussianBlur(image, (0, 0), 0.8)
    image = cv2.addWeighted(image, 1.14, blur, -0.14, 0)

    background = _detect_background_color(image)
    bg_fill = _hex_from_bgr(background)

    # Detalhes escuros são processados separado.
    dark_mask = _dark_detail_mask(image)
    dark_color = _dominant_color_from_mask(image, dark_mask)
    dark_fill = _hex_from_bgr(dark_color)

    # Remove detalhes escuros do k-means para não embolar texto.
    color_image = image.copy()
    color_image[dark_mask > 0] = background

    k = 8 if quality == "economico" else 11 if quality == "balanceado" else 14
    pixels = color_image.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60 if quality == "ultra" else 45, 0.7)

    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)
    labels = labels.flatten().reshape((height, width))

    min_area = max(4 if quality == "ultra" else 7, int(total_area * (0.000012 if quality == "ultra" else 0.00002)))

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<metadata>CDRvetor AI V6 - dark detail separated, quality={quality}, transparent={transparent}</metadata>'
    ]

    if not transparent:
        svg_parts.append(f'<rect id="background" width="100%" height="100%" fill="{bg_fill}"/>')

    color_order = []
    for i, color in enumerate(centers):
        color_order.append((int(np.sum(labels == i)), i, color))

    color_order.sort(reverse=True)

    for _, color_index, color in color_order:
        if _is_near_white(color) or _color_distance(color, background) < 18:
            continue

        fill = _hex_from_bgr(color)
        mask = (labels == color_index).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(dark_mask))
        mask = _clean_color_mask(mask, min_area, quality=quality)

        _add_paths_from_mask(svg_parts, mask, fill, min_area, quality=quality, force_polygon=False, group_name="camada_cor")

    detail_min_area = max(3, int(total_area * 0.000008))

    # Texto/detalhes por último, por cima.
    _add_paths_from_mask(svg_parts, dark_mask, dark_fill, detail_min_area, quality=quality, force_polygon=True, group_name="camada_texto_detalhe")

    svg_parts.append("</svg>")

    Path(output_path).write_text("\n".join(svg_parts), encoding="utf-8")
