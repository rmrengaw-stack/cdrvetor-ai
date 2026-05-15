import cv2
import numpy as np
from pathlib import Path


def _hex_from_bgr(color):
    b, g, r = [int(v) for v in color]
    return f"#{r:02x}{g:02x}{b:02x}"


def _contour_to_smooth_path(contour, smoothness=0.18):
    """
    Converte pontos de contorno em curvas Bézier cúbicas.
    Resultado: SVG mais suave e com menos aparência "quadrada".
    """
    points = contour.reshape(-1, 2).astype(float)

    if len(points) < 4:
        return ""

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


def _remove_tiny_islands(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)

    return clean


def vectorize_image(input_path: str, output_path: str):
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Não foi possível abrir a imagem.")

    height, width = image.shape[:2]
    total_area = width * height

    # 1) Pré-processamento: reduz ruído sem destruir bordas
    image = cv2.bilateralFilter(image, 9, 65, 65)

    # 2) Leve sharpen para recuperar bordas
    blur = cv2.GaussianBlur(image, (0, 0), 1.2)
    image = cv2.addWeighted(image, 1.35, blur, -0.35, 0)

    # 3) Redução de paleta
    k = 10
    pixels = image.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        35,
        0.8
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    labels = labels.flatten().reshape((height, width))

    min_area = max(20, int(total_area * 0.00006))

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<metadata>CDRvetor AI V2 - smooth color vectorization</metadata>',
        '<rect id="background" width="100%" height="100%" fill="white"/>'
    ]

    # 4) Desenha cores maiores primeiro
    color_order = []
    for i, color in enumerate(centers):
        count = int(np.sum(labels == i))
        color_order.append((count, i, color))

    color_order.sort(reverse=True)

    for _, color_index, color in color_order:
        fill = _hex_from_bgr(color)

        mask = (labels == color_index).astype(np.uint8) * 255

        # 5) Limpeza de manchas pequenas
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = _remove_tiny_islands(mask, min_area)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        svg_parts.append(
            f'<g id="camada_cor_{fill.replace("#", "")}" fill="{fill}" stroke="none">'
        )

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            epsilon = max(1.0, 0.0028 * perimeter)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 4:
                continue

            d = _contour_to_smooth_path(approx, smoothness=0.18)

            if d:
                svg_parts.append(f'<path d="{d}"/>')

        svg_parts.append("</g>")

    svg_parts.append("</svg>")

    Path(output_path).write_text("\n".join(svg_parts), encoding="utf-8")
