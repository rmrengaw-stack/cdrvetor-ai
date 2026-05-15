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
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid

from vectorizer import vectorize_image

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="CDRvetor AI")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/vectorize")
async def vectorize(
    file: UploadFile = File(...),
    quality: str = Form("balanceado"),
    transparent: str = Form("false")
):
    ext = Path(file.filename or "upload.png").suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        raise HTTPException(status_code=400, detail="Formato não suportado. Use PNG, JPG, WEBP ou BMP.")

    job_id = str(uuid.uuid4())
    input_path = INPUT_DIR / f"{job_id}{ext}"
    output_path = OUTPUT_DIR / f"{job_id}.svg"

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transparent_bool = str(transparent).lower() in ["true", "1", "sim", "yes", "on"]

    try:
        vectorize_image(
            str(input_path),
            str(output_path),
            quality=quality,
            transparent=transparent_bool
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao vetorizar: {e}")

    return FileResponse(
        output_path,
        media_type="image/svg+xml",
        filename="cdrvetor-ai.svg"
    )
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CDRvetor AI</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #101014;
      color: #f4f4f5;
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
    }
    .card {
      width: min(820px, 92vw);
      background: #18181f;
      border: 1px solid #2b2b35;
      border-radius: 24px;
      padding: 32px;
      box-shadow: 0 24px 80px rgba(0,0,0,.35);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 38px;
    }
    p {
      color: #b8b8c3;
      line-height: 1.5;
    }
    .drop {
      border: 2px dashed #555567;
      border-radius: 20px;
      padding: 28px;
      text-align: center;
      margin: 24px 0;
      background: #131318;
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin: 20px 0;
    }
    label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
    }
    select {
      width: 100%;
      border: 1px solid #3a3a46;
      background: #111118;
      color: white;
      border-radius: 12px;
      padding: 12px;
    }
    .check {
      display: flex;
      gap: 10px;
      align-items: center;
      background: #111118;
      border: 1px solid #3a3a46;
      border-radius: 12px;
      padding: 12px;
    }
    button, a.button {
      display: inline-block;
      background: #ffffff;
      color: #111;
      border: 0;
      border-radius: 14px;
      padding: 14px 22px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      margin-top: 12px;
    }
    button:disabled {
      opacity: .45;
      cursor: not-allowed;
    }
    .preview {
      margin-top: 24px;
      display: grid;
      gap: 16px;
    }
    iframe {
      width: 100%;
      height: 420px;
      background: white;
      border: 0;
      border-radius: 18px;
    }
    .status {
      min-height: 24px;
      color: #d7d7df;
    }
    .badge {
      display: inline-block;
      background: #243f2e;
      color: #bdf7d0;
      border: 1px solid #3d7a51;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      margin-bottom: 12px;
    }
  </style>
</head>
<body>
  <main class="card">
    <span class="badge">V5 Profissional</span>
    <h1>CDRvetor AI</h1>
    <p>Envie uma imagem e baixe um SVG vetorizado com modos de qualidade e opção de fundo transparente.</p>

    <div class="drop">
      <strong>Selecione PNG, JPG, WEBP ou BMP</strong><br>
      <input id="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" />
    </div>

    <div class="controls">
      <div>
        <label for="quality">Qualidade</label>
        <select id="quality">
          <option value="economico">Econômico — SVG mais leve</option>
          <option value="balanceado" selected>Balanceado — recomendado</option>
          <option value="ultra">Ultra — mais detalhes</option>
        </select>
      </div>

      <div>
        <label>Fundo</label>
        <div class="check">
          <input id="transparent" type="checkbox" />
          <span>Remover fundo / SVG transparente</span>
        </div>
      </div>
    </div>

    <button id="btn">Vetorizar agora</button>
    <p class="status" id="status"></p>

    <div class="preview" id="preview" style="display:none">
      <h3>Prévia do SVG</h3>
      <iframe id="frame"></iframe>
      <a class="button" id="download" download="cdrvetor-ai.svg">Baixar SVG</a>
    </div>
  </main>

  <script>
    const fileInput = document.getElementById("file");
    const btn = document.getElementById("btn");
    const statusEl = document.getElementById("status");
    const preview = document.getElementById("preview");
    const frame = document.getElementById("frame");
    const download = document.getElementById("download");
    const quality = document.getElementById("quality");
    const transparent = document.getElementById("transparent");

    btn.addEventListener("click", async () => {
      const file = fileInput.files[0];

      if (!file) {
        statusEl.textContent = "Escolha uma imagem primeiro.";
        return;
      }

      btn.disabled = true;
      statusEl.textContent = "Vetorizando...";

      const formData = new FormData();
      formData.append("file", file);
      formData.append("quality", quality.value);
      formData.append("transparent", transparent.checked ? "true" : "false");

      try {
        const response = await fetch("/vectorize", {
          method: "POST",
          body: formData
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Erro ao processar.");
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        frame.src = url;
        download.href = url;
        preview.style.display = "grid";
        statusEl.textContent = "SVG pronto.";
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
