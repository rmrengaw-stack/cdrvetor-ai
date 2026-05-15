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


def _bezier_path(points, smoothness=0.14):
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


def _contour_to_path(contour, area, perimeter):
    """
    V4: menos agressivo em objetos pequenos/texto.
    Suaviza formas grandes e preserva cantos pequenos.
    """
    if area < 2200:
        epsilon_factor = 0.0008
        smoothness = 0.06
    elif area < 10000:
        epsilon_factor = 0.0013
        smoothness = 0.10
    else:
        epsilon_factor = 0.0020
        smoothness = 0.14

    epsilon = max(0.35, epsilon_factor * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2).astype(float)

    if len(points) < 3:
        return ""

    if area < 2600:
        return _polygon_path(points)

    return _bezier_path(points, smoothness=smoothness)


def _is_near_white(color, threshold=242):
    b, g, r = [int(v) for v in color]
    return r > threshold and g > threshold and b > threshold


def _color_distance(c1, c2):
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    return float(np.linalg.norm(c1 - c2))


def _detect_background_color(image):
    """
    Estima cor de fundo usando as bordas da imagem.
    Ajuda a não transformar fundo em objeto grande.
    """
    h, w = image.shape[:2]
    border = np.concatenate([
        image[0:8, :, :].reshape(-1, 3),
        image[h-8:h, :, :].reshape(-1, 3),
        image[:, 0:8, :].reshape(-1, 3),
        image[:, w-8:w, :].reshape(-1, 3),
    ], axis=0)

    return np.median(border, axis=0).astype(np.uint8)


def _protect_dark_details(image, labels, centers):
    """
    V4: cria uma máscara especial para detalhes escuros.
    Isso ajuda números e letras como o "05" a não perderem furos.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detalhes escuros: threshold automático.
    _, dark = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Mantém só regiões realmente escuras.
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    # Acha cor dominante dos pixels escuros e força uma camada separada.
    dark_pixels = image[dark > 0]
    if len(dark_pixels) < 20:
        return labels, centers

    dark_color = np.median(dark_pixels, axis=0).astype(np.uint8)

    # Substitui o centro menos usado por cor escura protegida.
    counts = [int(np.sum(labels == i)) for i in range(len(centers))]
    replace_index = int(np.argmin(counts))
    centers[replace_index] = dark_color

    labels[dark > 0] = replace_index

    return labels, centers


def _clean_mask(mask, min_area, preserve_details=False):
    kernel = np.ones((2, 2), np.uint8)

    if preserve_details:
        # Não fecha demais letras e furos
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Edge-aware: usa Canny para recuperar bordas importantes
    edges = cv2.Canny(mask, 50, 140)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.bitwise_or(mask, edges)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)

    return clean


def _add_paths_from_mask(svg_parts, mask, fill, min_area, preserve_details=False):
    """
    Usa RETR_CCOMP + fill-rule evenodd para furos internos.
    """
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
        d = _contour_to_path(contour, area, perimeter)

        if not d:
            continue

        used.add(idx)

        child = hierarchy[idx][2]
        while child != -1:
            hole = contours[child]
            hole_area = cv2.contourArea(hole)

            # V4: preserva furos menores em texto/números.
            hole_limit = max(3, min_area * (0.18 if preserve_details else 0.30))

            if hole_area >= hole_limit:
                hole_perimeter = cv2.arcLength(hole, True)
                hole_d = _contour_to_path(hole, hole_area, hole_perimeter)
                if hole_d:
                    d += " " + hole_d

            used.add(child)
            child = hierarchy[child][0]

        svg_parts.append(f'<path d="{d}"/>')

    svg_parts.append("</g>")


def vectorize_image(input_path: str, output_path: str):
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Não foi possível abrir a imagem.")

    height, width = image.shape[:2]
    total_area = width * height

    # Pré-processamento conservador.
    image = cv2.bilateralFilter(image, 7, 42, 42)

    # Sharpen leve.
    blur = cv2.GaussianBlur(image, (0, 0), 0.9)
    image = cv2.addWeighted(image, 1.18, blur, -0.18, 0)

    background = _detect_background_color(image)

    # Mais cores para proteger texto/contornos.
    k = 14
    pixels = image.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        45,
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

    # Proteção especial para letras/números escuros.
    labels, centers = _protect_dark_details(image, labels, centers)

    min_area = max(6, int(total_area * 0.000018))

    bg_fill = _hex_from_bgr(background)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<metadata>CDRvetor AI V4 - edge-aware detail protection</metadata>',
        f'<rect id="background" width="100%" height="100%" fill="{bg_fill}"/>'
    ]

    color_order = []
    for i, color in enumerate(centers):
        count = int(np.sum(labels == i))
        color_order.append((count, i, color))

    color_order.sort(reverse=True)

    for _, color_index, color in color_order:
        # Ignora branco e também cor muito parecida com o fundo.
        if _is_near_white(color) or _color_distance(color, background) < 18:
            continue

        fill = _hex_from_bgr(color)
        preserve_details = np.mean(color) < 120

        mask = (labels == color_index).astype(np.uint8) * 255
        mask = _clean_mask(mask, min_area, preserve_details=preserve_details)

        _add_paths_from_mask(
            svg_parts,
            mask,
            fill,
            min_area,
            preserve_details=preserve_details
        )

    svg_parts.append("</svg>")

    Path(output_path).write_text("\n".join(svg_parts), encoding="utf-8")
