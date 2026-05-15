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


def _bezier_path(points, smoothness=0.16):
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
    V3: preserva detalhes pequenos e suaviza áreas grandes.
    Letras/números perdem menos detalhe.
    """
    # Objetos pequenos recebem menos simplificação
    if area < 2500:
        epsilon_factor = 0.0012
        smoothness = 0.08
    elif area < 12000:
        epsilon_factor = 0.0018
        smoothness = 0.12
    else:
        epsilon_factor = 0.0024
        smoothness = 0.16

    epsilon = max(0.45, epsilon_factor * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2).astype(float)

    if len(points) < 3:
        return ""

    # Contornos pequenos/texto: linhas preservam melhor cantos.
    if area < 1800:
        return _polygon_path(points)

    return _bezier_path(points, smoothness=smoothness)


def _clean_mask(mask, min_area):
    """
    Remove sujeiras muito pequenas, mas preserva detalhes úteis.
    """
    kernel = np.ones((2, 2), np.uint8)

    # Menos agressivo que a V2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)

    return clean


def _is_near_white(color, threshold=238):
    b, g, r = [int(v) for v in color]
    return r > threshold and g > threshold and b > threshold


def vectorize_image(input_path: str, output_path: str):
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Não foi possível abrir a imagem.")

    height, width = image.shape[:2]
    total_area = width * height

    # 1) Pré-processamento mais conservador
    image = cv2.bilateralFilter(image, 7, 45, 45)

    # 2) Sharpen leve
    blur = cv2.GaussianBlur(image, (0, 0), 1.0)
    image = cv2.addWeighted(image, 1.22, blur, -0.22, 0)

    # 3) Paleta um pouco maior para preservar detalhes de logos/textos
    k = 12
    pixels = image.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.7
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

    # V3: área mínima menor para não destruir números/letras.
    min_area = max(8, int(total_area * 0.000025))

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<metadata>CDRvetor AI V3 - detail preserving vectorization</metadata>',
        '<rect id="background" width="100%" height="100%" fill="white"/>'
    ]

    color_order = []
    for i, color in enumerate(centers):
        count = int(np.sum(labels == i))
        color_order.append((count, i, color))

    color_order.sort(reverse=True)

    for _, color_index, color in color_order:
        # Ignora camada quase branca para evitar massas brancas sobre o desenho
        if _is_near_white(color):
            continue

        fill = _hex_from_bgr(color)
        mask = (labels == color_index).astype(np.uint8) * 255
        mask = _clean_mask(mask, min_area)

        # V3: RETR_CCOMP preserva furos internos.
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None or not contours:
            continue

        hierarchy = hierarchy[0]

        svg_parts.append(
            f'<g id="camada_cor_{fill.replace("#", "")}" fill="{fill}" stroke="none" fill-rule="evenodd">'
        )

        # Agrupa contorno pai com seus furos internos.
        used = set()

        for idx, contour in sorted(
            enumerate(contours),
            key=lambda item: cv2.contourArea(item[1]),
            reverse=True
        ):
            if idx in used:
                continue

            parent = hierarchy[idx][3]

            # Só começa por contornos externos
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

            # Adiciona furos filhos no mesmo path.
            child = hierarchy[idx][2]
            while child != -1:
                hole = contours[child]
                hole_area = cv2.contourArea(hole)

                # Furos pequenos são importantes em letras/números.
                if hole_area >= max(5, min_area * 0.35):
                    hole_perimeter = cv2.arcLength(hole, True)
                    hole_d = _contour_to_path(hole, hole_area, hole_perimeter)
                    if hole_d:
                        d += " " + hole_d

                used.add(child)
                child = hierarchy[child][0]

            svg_parts.append(f'<path d="{d}"/>')

        svg_parts.append("</g>")

    svg_parts.append("</svg>")

    Path(output_path).write_text("\n".join(svg_parts), encoding="utf-8")
