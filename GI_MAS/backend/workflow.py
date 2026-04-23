from __future__ import annotations

import asyncio
from io import BytesIO
import hashlib
from pathlib import Path
import random
from typing import Awaitable, Callable
from urllib.parse import quote_plus

import httpx
from PIL import Image, ImageDraw, ImageFilter, ImageStat, UnidentifiedImageError

from agents.brand_agent import build_brand_direction
from agents.critic_agent import review_and_improve_prompts
from agents.planner_agent import create_visual_plan
from agents.prompt_designer_agent import design_five_prompts
from core.config import (
    GROQ_VISION_GUARD,
    GROQ_VISION_MODEL,
    IMAGE_CONCURRENCY,
    IMAGE_HEIGHT,
    IMAGE_MODEL,
    IMAGE_WIDTH,
    IMAGE_REQUEST_DELAY,
)
from core.groq_client import ask_groq_vision_json

BASE_DIR = Path(__file__).resolve().parents[1]
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ProgressCallback = Callable[[str, str, str], Awaitable[None]]


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


def _build_image_url(prompt: str, seed: int) -> str:
    encoded = quote_plus(prompt)
    return (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?model={IMAGE_MODEL}&width={IMAGE_WIDTH}&height={IMAGE_HEIGHT}&seed={seed}"
    )


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    raw = color.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return (15, 140, 149)
    try:
        return tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (15, 140, 149)


def _enforce_visual_style(prompt: str) -> str:
    return (
        f"{prompt} "
        "No people, no text, no watermark."
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _extract_palette_from_file(image_path: str | None, max_colors: int = 3) -> list[str]:
    if not image_path:
        return []
    try:
        img = Image.open(image_path).convert("RGB").resize((160, 160), Image.LANCZOS)
        pal = img.quantize(colors=max_colors, method=Image.MEDIANCUT).convert("RGB")
        # getcolors: list[(count, (r,g,b))]
        color_counts = pal.getcolors(maxcolors=256) or []
        color_counts.sort(reverse=True, key=lambda x: x[0])
        return [_rgb_to_hex(c[1]) for c in color_counts[:max_colors]]
    except Exception:
        return []


def _describe_advisor_style(advisor_image_path: str | None) -> str:
    if not advisor_image_path:
        return "professional corporate portrait look"
    try:
        img = Image.open(advisor_image_path).convert("RGB").resize((200, 200), Image.LANCZOS)
        stat = ImageStat.Stat(img)
        r, g, b = stat.mean
        brightness = sum((r, g, b)) / 3
        warm = (r + 10) > b
        tone = "warm" if warm else "cool"
        light = "bright" if brightness > 140 else "moody"
        return f"{tone} {light} executive portrait look"
    except Exception:
        return "professional corporate portrait look"


async def _background_has_extra_people(background_bytes: bytes) -> tuple[bool, str]:
    if not GROQ_VISION_GUARD:
        return False, "vision_guard_disabled"

    prompt = (
        "Inspect this background image only. "
        "Return JSON with keys: extra_people (boolean), people_count (integer), reason (string). "
        "Mark extra_people=true if any visible human/person/face/silhouette is present."
    )
    try:
        result = await ask_groq_vision_json(background_bytes, prompt, model=GROQ_VISION_MODEL)
    except Exception as exc:
        return False, f"vision_guard_error:{type(exc).__name__}"

    value = result.get("extra_people")
    if isinstance(value, bool):
        return value, str(result.get("reason", ""))

    raw = str(result.get("raw", "")).lower()
    if "true" in raw:
        return True, "vision_raw_true"
    return False, "vision_raw_false"


async def _download_image(remote_url: str, max_retries: int = 5) -> bytes:
    """Download image with exponential backoff retry."""
    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.get(remote_url)
                if response.status_code == 429 and attempt < max_retries:
                    wait_time = 2 ** (attempt - 1) + (attempt * 0.5)
                    await asyncio.sleep(wait_time)
                    continue
                response.raise_for_status()
                return response.content
            except Exception as exc:
                if attempt < max_retries:
                    wait_time = 2 ** (attempt - 1) + (attempt * 0.5)
                    await asyncio.sleep(wait_time)
                    continue
                raise
    raise RuntimeError("Image download failed after retries")


def _create_fallback_background_bytes(brand_colors: list[str], seed: int, variant: int) -> bytes:
    rng = random.Random(seed * 97 + variant * 101)
    rgb_colors = [_hex_to_rgb(c) for c in brand_colors] or [(15, 140, 149), (18, 180, 217), (230, 244, 255)]

    top = rgb_colors[(variant + seed) % len(rgb_colors)]
    bottom = rgb_colors[(variant + seed + 1) % len(rgb_colors)]

    base = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), top)
    draw = ImageDraw.Draw(base)

    for y in range(IMAGE_HEIGHT):
        t = y / max(1, IMAGE_HEIGHT - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        draw.line([(0, y), (IMAGE_WIDTH, y)], fill=(r, g, b))

    # Soft light layers for depth without hard geometric patterns.
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)
    for _ in range(4):
        cx = rng.randint(-200, IMAGE_WIDTH + 200)
        cy = rng.randint(-120, IMAGE_HEIGHT + 120)
        r = rng.randint(220, 520)
        alpha = rng.randint(18, 42)
        odraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, alpha))

    merged = Image.alpha_composite(base.convert("RGBA"), overlay)
    merged = merged.filter(ImageFilter.GaussianBlur(radius=8)).convert("RGB")

    buf = BytesIO()
    merged.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _apply_brand_stripes(base: Image.Image, brand_colors: list[str]) -> Image.Image:
    w, h = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rgb_colors = [_hex_to_rgb(c) for c in brand_colors] or [(15, 140, 149)]

    top_height = max(40, int(h * 0.1))
    bottom_height = max(46, int(h * 0.12))

    for idx, rgb in enumerate(rgb_colors[:3]):
        draw.rectangle(
            [0, idx * 14, w, idx * 14 + top_height],
            fill=(rgb[0], rgb[1], rgb[2], 120),
        )

    for idx, rgb in enumerate(rgb_colors[:3]):
        y_start = h - bottom_height - idx * 16
        draw.rectangle(
            [0, y_start, w, y_start + bottom_height],
            fill=(rgb[0], rgb[1], rgb[2], 110),
        )

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def _open_overlay_image(image_path: str, asset_name: str) -> tuple[Image.Image | None, str | None]:
    suffix = Path(image_path).suffix.lower()

    # Pillow SVG acamaz. Cairosvg kuruluysa donusturup kullanmayi deneriz.
    if suffix == ".svg":
        try:
            import cairosvg  # type: ignore

            png_bytes = cairosvg.svg2png(url=image_path)
            overlay = Image.open(BytesIO(png_bytes)).convert("RGBA")
            return overlay, None
        except Exception:
            warning = (
                f"{asset_name} SVG formatinda yuklendi. Bu ortamda SVG islenemedigi icin atlandi. "
                "Logo/danisman gorselini PNG veya JPG olarak yukleyin."
            )
            return None, warning

    try:
        overlay = Image.open(image_path).convert("RGBA")
        return overlay, None
    except UnidentifiedImageError:
        warning = (
            f"{asset_name} dosyasi acilamadi ve atlandi. Lütfen PNG veya JPG formatinda yukleyin."
        )
        return None, warning


def _extract_person_cutout(advisor: Image.Image) -> tuple[Image.Image, str | None]:
    def _crop_alpha(img: Image.Image) -> Image.Image:
        alpha = img.split()[-1]
        bbox = alpha.getbbox()
        return img.crop(bbox) if bbox else img

    def _foreground_ratio(img: Image.Image) -> float:
        alpha = img.split()[-1]
        vals = list(alpha.getdata())
        total = len(vals) if vals else 1
        fg = sum(1 for v in vals if v > 20)
        return fg / total

    def _has_soft_alpha(img: Image.Image) -> bool:
        alpha = img.split()[-1]
        extrema = alpha.getextrema()
        return extrema is not None and extrema[0] < 20 and extrema[1] > 230

    def _refine_single_subject(img: Image.Image) -> Image.Image:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            rgba = np.array(img.convert("RGBA"))
            alpha = rgba[:, :, 3]
            mask = (alpha > 30).astype("uint8")
            if mask.sum() == 0:
                return img

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels <= 1:
                return img

            h, w = mask.shape
            cx_ref, cy_ref = w * 0.5, h * 0.58
            best_label = 1
            best_score = -1e18
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                cx, cy = centroids[label]
                dist = ((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2) ** 0.5
                score = area - dist * 30.0
                if score > best_score:
                    best_score = score
                    best_label = label

            main_mask = (labels == best_label).astype("uint8") * 255
            main_mask = cv2.GaussianBlur(main_mask, (5, 5), 0)
            rgba[:, :, 3] = main_mask
            return Image.fromarray(rgba, "RGBA")
        except Exception:
            return img

    def _grabcut_fallback(image: Image.Image) -> Image.Image | None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            rgb = np.array(image.convert("RGB"))
            h, w = rgb.shape[:2]
            rect = (max(1, int(w * 0.12)), max(1, int(h * 0.06)), int(w * 0.76), int(h * 0.9))
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            mask2 = ((mask == 1) | (mask == 3)).astype("uint8")
            if mask2.sum() < int(w * h * 0.03):
                return None

            alpha = (mask2 * 255).astype("uint8")
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            rgba = np.dstack([rgb, alpha])
            return Image.fromarray(rgba, "RGBA")
        except Exception:
            return None

    def _simple_color_key_fallback(image: Image.Image) -> Image.Image | None:
        # OpenCV/rembg olmadan da calissin diye basit arka plan ayirma fallback'i.
        try:
            rgba = image.convert("RGBA")
            w, h = rgba.size
            pix = rgba.load()

            sample_points = [
                (0, 0),
                (w - 1, 0),
                (0, h - 1),
                (w - 1, h - 1),
                (w // 2, 0),
                (w // 2, h - 1),
            ]
            sr = sg = sb = 0
            for x, y in sample_points:
                r, g, b, _ = pix[x, y]
                sr += r
                sg += g
                sb += b
            n = len(sample_points)
            br, bg, bb = sr // n, sg // n, sb // n

            threshold = 62
            out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            out_pix = out.load()

            for y in range(h):
                for x in range(w):
                    r, g, b, _ = pix[x, y]
                    dist = abs(r - br) + abs(g - bg) + abs(b - bb)
                    if dist > threshold:
                        out_pix[x, y] = (r, g, b, 255)
                    else:
                        out_pix[x, y] = (r, g, b, 0)

            alpha = out.split()[-1]
            alpha = alpha.filter(ImageFilter.GaussianBlur(1.6))
            out.putalpha(alpha)
            return out
        except Exception:
            return None

    try:
        from rembg import remove  # type: ignore

        src_buf = BytesIO()
        advisor.convert("RGBA").save(src_buf, format="PNG")
        cutout_png = remove(src_buf.getvalue())
        cutout = _refine_single_subject(Image.open(BytesIO(cutout_png)).convert("RGBA"))
        ratio = _foreground_ratio(cutout)
        if _has_soft_alpha(cutout) and 0.12 <= ratio <= 0.88:
            return _crop_alpha(cutout), None
        rembg_error_name = "OpaqueMask"
    except Exception as exc:
        rembg_error_name = type(exc).__name__

    # rembg sonucu yetersizse fallback segmentasyon dene
    fallback = _grabcut_fallback(advisor)
    if fallback is not None:
        fallback = _refine_single_subject(fallback)
        ratio = _foreground_ratio(fallback)
        if 0.12 <= ratio <= 0.9:
            return _crop_alpha(fallback), None

    # Son fallback: PIL tabanli renk anahtarlama
    simple = _simple_color_key_fallback(advisor)
    if simple is not None:
        simple = _refine_single_subject(simple)
        ratio = _foreground_ratio(simple)
        if 0.12 <= ratio <= 0.9:
            return _crop_alpha(simple), None

    return advisor, f"Danisman arka plan ayirma basarisiz: {rembg_error_name}. Danisman gorseli eklenmedi."


def _paste_advisor(base: Image.Image, advisor_image_path: str | None) -> tuple[Image.Image, str | None]:
    if not advisor_image_path:
        return base, None

    advisor, warning = _open_overlay_image(advisor_image_path, "Danisman gorseli")
    if advisor is None:
        return base, warning

    advisor, seg_warning = _extract_person_cutout(advisor)
    if seg_warning:
        # Kesim basarisizsa dikdortgen resmi asla yapistirma.
        return base, seg_warning

    w, h = base.size
    target_h = int(h * 0.82)
    ratio = target_h / advisor.height
    target_w = max(160, int(advisor.width * ratio))
    advisor = advisor.resize((target_w, target_h), Image.LANCZOS)

    shadow = Image.new("RGBA", advisor.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rectangle([18, 18, advisor.width, advisor.height], fill=(0, 0, 0, 120))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))

    base_rgba = base.convert("RGBA")
    x = int(w * 0.04)
    y = h - advisor.height - int(h * 0.05)
    base_rgba.paste(shadow, (x, y), shadow)
    base_rgba.paste(advisor, (x, y), advisor)
    return base_rgba.convert("RGB"), seg_warning


def _paste_logo(base: Image.Image, logo_image_path: str | None) -> tuple[Image.Image, str | None]:
    if not logo_image_path:
        return base, None

    logo, warning = _open_overlay_image(logo_image_path, "Logo")
    if logo is None:
        return base, warning

    w, h = base.size
    target_w = int(w * 0.2)
    ratio = target_w / logo.width
    target_h = max(48, int(logo.height * ratio))
    logo = logo.resize((target_w, target_h), Image.LANCZOS)

    panel_pad = 4
    panel = Image.new("RGBA", (logo.width + panel_pad * 2, logo.height + panel_pad * 2), (255, 255, 255, 140))
    panel.paste(logo, (panel_pad, panel_pad), logo)

    base_rgba = base.convert("RGBA")
    x = w - panel.width - int(w * 0.03)
    y = int(h * 0.04)
    base_rgba.paste(panel, (x, y), panel)
    return base_rgba.convert("RGB"), None


def _compose_with_assets(
    background_bytes: bytes,
    advisor_image_path: str | None,
    logo_image_path: str | None,
    brand_colors: list[str],
) -> tuple[bytes, list[str]]:
    image = Image.open(BytesIO(background_bytes)).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
    image = _apply_brand_stripes(image, brand_colors)
    warnings: list[str] = []
    image, advisor_warning = _paste_advisor(image, advisor_image_path)
    if advisor_warning:
        warnings.append(advisor_warning)

    image, logo_warning = _paste_logo(image, logo_image_path)
    if logo_warning:
        warnings.append(logo_warning)

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), warnings


def _image_metrics(image_bytes: bytes) -> dict:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean_luma = round(stat.mean[0], 2)
    contrast = round(stat.stddev[0], 2)
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edge)
    edge_intensity = round(edge_stat.mean[0], 2)

    return {
        "brightness": mean_luma,
        "contrast": contrast,
        "edge_intensity": edge_intensity,
    }


async def run_multi_agent_image_pipeline(
    user_prompt: str,
    institution_name: str,
    advisor_profile: str,
    brand_colors: list[str],
    advisor_image_path: str | None = None,
    logo_image_path: str | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict:
    # Background-only mode: advisor image compositing disabled.
    advisor_image_path = None

    warnings: list[str] = []

    logo_palette = _extract_palette_from_file(logo_image_path, max_colors=3)
    advisor_palette: list[str] = []
    advisor_style = "empty-stage background for later person placement"

    enriched_colors = list(brand_colors)
    for c in logo_palette + advisor_palette:
        if c not in enriched_colors:
            enriched_colors.append(c)

    if progress_cb:
        await progress_cb("planner", "in_progress", "Planner analyzing brief")

    try:
        visual_plan = await create_visual_plan(user_prompt, institution_name, advisor_profile)
    except Exception as exc:
        visual_plan = (
            "Generate 5 distinct corporate background variations. "
            "Balance brand colors evenly. "
            "Keep a clean area with no people for later person placement."
        )
        reason = "429 rate limit" if _is_rate_limit_error(exc) else type(exc).__name__
        warnings.append(f"Planner fallback used: {reason}")

    if progress_cb:
        await progress_cb("planner", "completed", "Planning completed")
        await progress_cb("brand_agent", "in_progress", "Brand agent extracting colors and tone")

    try:
        brand_direction = await build_brand_direction(institution_name, brand_colors)
    except Exception as exc:
        color_text = ", ".join(enriched_colors) if enriched_colors else "#0E3A8A, #00A8E8, #E6F4FF"
        brand_direction = (
            "Professional, trustworthy and modern tone. "
            f"Brand colors: {color_text}. "
            "High contrast readable layers with clean composition."
        )
        reason = "429 rate limit" if _is_rate_limit_error(exc) else type(exc).__name__
        warnings.append(f"Brand agent fallback used: {reason}")

    if progress_cb:
        await progress_cb("brand_agent", "completed", "Brand direction prepared")
        await progress_cb("prompt_designer", "in_progress", "Prompt designer creating 5 concepts")

    try:
        designed_prompts = await design_five_prompts(
            user_prompt=user_prompt,
            institution_name=institution_name,
            advisor_profile=f"{advisor_profile}. Visual style hint: {advisor_style}.",
            brand_colors=enriched_colors,
            visual_plan=visual_plan,
            brand_direction=(
                f"{brand_direction}\n"
                f"Logo palette: {', '.join(logo_palette) if logo_palette else 'none'}\n"
                f"Advisor palette: {', '.join(advisor_palette) if advisor_palette else 'none'}\n"
                f"Advisor style: {advisor_style}"
            ),
        )
    except Exception as exc:
        color_text = ", ".join(enriched_colors)
        designed_prompts = [
            {
                "title": f"Concept {i}",
                "style_note": "Corporate and clean",
                "prompt": (
                    f"Corporate campaign background for {institution_name}. "
                    f"Theme: {user_prompt}. Brand colors: {color_text}. "
                    "No people. Keep clean space for later subject placement. "
                    f"Variation {i}."
                ),
            }
            for i in range(1, 6)
        ]
        reason = "429 rate limit" if _is_rate_limit_error(exc) else type(exc).__name__
        warnings.append(f"Prompt designer fallback used: {reason}")

    if progress_cb:
        await progress_cb("prompt_designer", "completed", "5 prompts designed")
        await progress_cb("critic", "in_progress", "Quality control improving prompts")

    try:
        reviewed_prompts = await review_and_improve_prompts(designed_prompts, enriched_colors)
    except Exception as exc:
        reviewed_prompts = designed_prompts
        reason = "429 rate limit" if _is_rate_limit_error(exc) else type(exc).__name__
        warnings.append(f"Critic step skipped, fallback prompts used: {reason}")

    if progress_cb:
        await progress_cb("critic", "completed", "Quality control completed")
        await progress_cb("segmenter", "completed", "Segmentation skipped in this mode")
        for i in range(1, 6):
            await progress_cb(f"segmenter.image_{i}", "completed", "Skipped")
        await progress_cb("renderer", "in_progress", "Generating images")
        await progress_cb("composer", "in_progress", "Starting compositing")

    semaphore = asyncio.Semaphore(max(1, IMAGE_CONCURRENCY))

    async def _process_single_image(index: int, prompt_obj: dict) -> tuple[dict | None, dict | None, list[str]]:
        local_warnings: list[str] = []
        try:
            async with semaphore:
                # Add delay between requests to avoid rate limiting
                if index > 1:
                    await asyncio.sleep(IMAGE_REQUEST_DELAY)
                
                if progress_cb:
                    await progress_cb(f"renderer.image_{index}", "in_progress", f"Image {index} rendering")

                final_prompt = _enforce_visual_style(prompt_obj["prompt"])

                try:
                    image_url = _build_image_url(final_prompt, seed=index)
                    background_bytes = await _download_image(image_url)
                except Exception as render_exc:
                    local_warnings.append(
                        f"Image {index} AI render failed, fallback used: {type(render_exc).__name__}"
                    )
                    background_bytes = _create_fallback_background_bytes(
                        enriched_colors,
                        seed=index + len(final_prompt),
                        variant=index,
                    )

                if progress_cb:
                    await progress_cb(f"composer.image_{index}", "in_progress", f"Image {index} compositing")

                composed_bytes, compose_warnings = _compose_with_assets(
                    background_bytes=background_bytes,
                    advisor_image_path=advisor_image_path,
                    logo_image_path=logo_image_path,
                    brand_colors=enriched_colors,
                )
                local_warnings.extend(compose_warnings)

                prompt_hash = hashlib.md5(final_prompt.encode("utf-8")).hexdigest()[:12]
                file_name = f"image_{index}_{prompt_hash}.jpg"
                target_path = IMAGES_DIR / file_name
                target_path.write_bytes(composed_bytes)

                image_obj = {
                    "id": index,
                    "title": prompt_obj["title"],
                    "style_note": prompt_obj["style_note"],
                    "prompt": final_prompt,
                    "image_url": f"/images/{file_name}",
                }

                if progress_cb:
                    await progress_cb(f"renderer.image_{index}", "completed", f"Image {index} rendered")
                    await progress_cb(f"composer.image_{index}", "completed", f"Image {index} saved")

                return image_obj, None, local_warnings
        except Exception as exc:
            local_warnings.append(f"Image {index} generation failed: {str(exc)}")
            if progress_cb:
                await progress_cb(f"renderer.image_{index}", "failed", f"Image {index} error")
                await progress_cb(f"composer.image_{index}", "failed", f"Image {index} compositing error")
            return None, None, local_warnings

    tasks = [
        asyncio.create_task(_process_single_image(i, prompt_obj))
        for i, prompt_obj in enumerate(reviewed_prompts, start=1)
    ]
    results = await asyncio.gather(*tasks)

    images = []
    for image_obj, _, local_warnings in results:
        for warn in local_warnings:
            if warn not in warnings:
                warnings.append(warn)
        if image_obj:
            images.append(image_obj)

    images.sort(key=lambda item: item["id"])

    if progress_cb:
        await progress_cb("renderer", "completed", "All images rendered")
        await progress_cb("composer", "completed", "All compositing completed")

    return {
        "agent_outputs": {
            "planner": visual_plan,
            "brand_agent": brand_direction,
            "critic_summary": "Promptlar kalite kontrol ajanindan gecirildi.",
        },
        "warnings": warnings,
        "images": images,
    }
