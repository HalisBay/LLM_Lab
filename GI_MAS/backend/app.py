from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import GROQ_MODEL
from workflow import run_multi_agent_image_pipeline

app = FastAPI(title="GI_MAS API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
IMAGES_DIR = BASE_DIR / "images"
UPLOADS_DIR = BASE_DIR / "uploads"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, dict] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "groq_model": GROQ_MODEL}


def _default_tree() -> dict:
    return {
        "planner": {"status": "pending", "message": "Pending", "children": {}},
        "brand_agent": {"status": "pending", "message": "Pending", "children": {}},
        "prompt_designer": {"status": "pending", "message": "Pending", "children": {}},
        "critic": {"status": "pending", "message": "Pending", "children": {}},
        "segmenter": {
            "status": "pending",
            "message": "Pending",
            "children": {
                f"image_{i}": {"status": "pending", "message": "Pending", "children": {}}
                for i in range(1, 6)
            },
        },
        "renderer": {
            "status": "pending",
            "message": "Pending",
            "children": {
                f"image_{i}": {"status": "pending", "message": "Pending", "children": {}}
                for i in range(1, 6)
            },
        },
        "composer": {
            "status": "pending",
            "message": "Pending",
            "children": {
                f"image_{i}": {"status": "pending", "message": "Pending", "children": {}}
                for i in range(1, 6)
            },
        },
    }


def _set_tree_status(tree: dict, key_path: str, status: str, message: str) -> None:
    parts = key_path.split(".")
    node = tree.get(parts[0])
    if not node:
        return
    for part in parts[1:]:
        node = node["children"].get(part)
        if not node:
            return
    node["status"] = status
    node["message"] = message


async def _save_upload(file: UploadFile | None, prefix: str, job_id: str) -> str | None:
    if file is None:
        return None
    ext = Path(file.filename or "").suffix or ".png"
    target = UPLOADS_DIR / f"{prefix}_{job_id}{ext}"
    content = await file.read()
    target.write_bytes(content)
    return str(target)


def _split_colors(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


async def _run_job(
    job_id: str,
    user_prompt: str,
    institution_name: str,
    advisor_profile: str,
    brand_colors: list[str],
    advisor_image_path: str | None,
    logo_image_path: str | None,
) -> None:
    async def _progress(step: str, status: str, message: str) -> None:
        _set_tree_status(JOBS[job_id]["agent_tree"], step, status, message)

    try:
        result = await run_multi_agent_image_pipeline(
            user_prompt=user_prompt,
            institution_name=institution_name,
            advisor_profile=advisor_profile,
            brand_colors=brand_colors,
            advisor_image_path=advisor_image_path,
            logo_image_path=logo_image_path,
            progress_cb=_progress,
        )
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = {
            "groq": {
                "model": GROQ_MODEL,
                "note": "Groq bu projede coklu ajan metin orkestrasyonu icin kullaniliyor.",
            },
            "image_generation": {
                "provider": "Pollinations + local compositing",
                "note": "Danisman ve logo yuklenirse ciktigida dogrudan kompozit edilir.",
            },
            **result,
        }
    except Exception as exc:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)


@app.post("/generate-images")
async def generate_images(
    user_prompt: str = Form(...),
    institution_name: str = Form(...),
    advisor_profile: str = Form(""),
    brand_colors: str = Form(...),
    advisor_image: UploadFile | None = File(default=None),
    logo_image: UploadFile | None = File(default=None),
) -> dict:
    colors = _split_colors(brand_colors)
    if len(user_prompt.strip()) < 5:
        raise HTTPException(status_code=400, detail="user_prompt must be at least 5 characters")
    if len(institution_name.strip()) < 2:
        raise HTTPException(status_code=400, detail="institution_name must be at least 2 characters")
    if not colors:
        raise HTTPException(status_code=400, detail="At least one brand color must be provided")

    job_id = str(uuid.uuid4())
    advisor_path = await _save_upload(advisor_image, "advisor", job_id)
    logo_path = await _save_upload(logo_image, "logo", job_id)
    effective_profile = advisor_profile.strip() or "Person in uploaded image"

    JOBS[job_id] = {
        "status": "running",
        "error": None,
        "result": None,
        "agent_tree": _default_tree(),
    }

    asyncio.create_task(
        _run_job(
            job_id=job_id,
            user_prompt=user_prompt.strip(),
            institution_name=institution_name.strip(),
            advisor_profile=effective_profile,
            brand_colors=colors,
            advisor_image_path=advisor_path,
            logo_image_path=logo_path,
        )
    )

    return {"job_id": job_id, "status": "running"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "error": job["error"],
        "agent_tree": job["agent_tree"],
        "result": job["result"],
    }


app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
