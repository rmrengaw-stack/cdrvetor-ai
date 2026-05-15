import cv2
import numpy as np
from pathlib import Path


def _svg_path_from_contour(contour):
    points = contour.reshape(-1, 2)
    if len(points) < 3:
        return ""

    d = f"M {int(points[0][0])} {int(points[0][1])} "
    for x, y in points[1:]:
        d += f"L {int(x)} {int(y)} "
    d += "Z"
    return d


def vectorize_image(input_path: str, output_path: str):
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Não foi possível abrir a imagem.")

    height, width = image.shape[:2]

    # Suaviza ruído sem destruir tanto as bordas
    smooth = cv2.bilateralFilter(image, 7, 50, 50)

    # Reduz paleta para gerar camadas por cor
    pixels = smooth.reshape((-1, 3)).astype(np.float32)
    k = 8  # número inicial de cores
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    labels = labels.flatten()

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>'
    ]

    min_area = max(16, int((width * height) * 0.00005))

    # Desenha cores maiores primeiro
    color_areas = []
    for i, color in enumerate(centers):
        count = int(np.sum(labels == i))
        color_areas.append((count, i, color))

    color_areas.sort(reverse=True)

    for _, i, color in color_areas:
        mask = (labels.reshape((height, width)) == i).astype(np.uint8) * 255

        # Limpeza da máscara
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        b, g, r = [int(v) for v in color]
        fill = f"#{r:02x}{g:02x}{b:02x}"

        svg_parts.append(f'<g id="cor_{fill.replace("#", "")}" fill="{fill}" stroke="none">')

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            epsilon = 0.0035 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            d = _svg_path_from_contour(approx)
            if d:
                svg_parts.append(f'<path d="{d}"/>')

        svg_parts.append("</g>")

    svg_parts.append("</svg>")

    Path(output_path).write_text("\n".join(svg_parts), encoding="utf-8")
