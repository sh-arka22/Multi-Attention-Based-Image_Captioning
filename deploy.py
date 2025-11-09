"""
Minimal Modal harness to run train_multi_attention.py on remote GPU machines.

Usage:
1. Upload local dataset artifacts to the Modal volume:
     modal run deploy.py::upload_dataset
2. Launch remote training (defaults to all attention types, batch=256, epochs=20):
     modal run deploy.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ModuleNotFoundError:  # pragma: no cover - optional for local CLI use
    FastAPI = UploadFile = File = HTTPException = CORSMiddleware = None

import modal

app = modal.App("image-caption-attention-trainer")
volume = modal.Volume.from_name("caption-dataset-models", create_if_missing=True)

DATASET_DIR = Path("/data/dataset")
OUTPUT_DIR = Path("/data/models")
SUPPORTED_ATTENTION_TYPES = ("luong", "bahdanau", "scaled_dot", "multihead", "concatenate")

TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "tensorflow[and-cuda]>=2.13.0",
        "numpy>=1.24.0",
        "fastapi>=0.104.0",
        "python-multipart>=0.0.6",
        "pillow>=10.0.0",
        "keras-preprocessing>=1.1.2",
    )
    .add_local_file("train_multi_attention.py", remote_path="/root/train_multi_attention.py")
    .add_local_file("caption_core.py", remote_path="/root/caption_core.py")
)

UPLOAD_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .add_local_dir("results", remote_path="/local_results")
)


def _normalize_attention_list(raw: Optional[Iterable[str]], default: Iterable[str]) -> List[str]:
    if not raw:
        return list(default)
    cleaned = [att.strip().lower() for att in raw if att and att.strip()]
    if not cleaned or cleaned == ["all"]:
        return list(default)
    return cleaned


@app.function(image=UPLOAD_IMAGE, volumes={"/data": volume}, timeout=300)
def upload_dataset():
    """Copy local dataset artifacts (captions/features/tokenizer) into the Modal volume."""
    import shutil

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("captions.csv", "img_features.pkl", "tokenizer.pkl"):
        src = Path("/local_results") / name
        dst = DATASET_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing local file: results/{name}")
        print(f"📤 Uploading {src} -> {dst}")
        shutil.copy(src, dst)

    volume.commit()
    print("✅ Dataset upload complete.")


@app.function(
    image=TRAIN_IMAGE,
    volumes={"/data": volume},
    memory=32768,
    timeout=8 * 60 * 60,
    gpu="H100",
)
def run_training(
    attention_types: Optional[List[str]] = None,
    batch_size: int = 256,
    epochs: int = 20,
    learning_rate: float = 1e-3,
):
    """Execute train_multi_attention training loop on Modal infrastructure."""
    import sys

    sys.path.insert(0, "/root")
    import train_multi_attention as tma  # pylint: disable=import-error

    dataset = DATASET_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    attn = _normalize_attention_list(attention_types, tma.SUPPORTED_ATTENTION_TYPES)

    config = tma.TrainingConfig(
        captions_path=dataset / "captions.csv",
        features_path=dataset / "img_features.pkl",
        tokenizer_path=dataset / "tokenizer.pkl",
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        attention_types=attn,
    )

    print("🚀 Starting remote multi-attention training")
    print(f"  Dataset dir: {dataset}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Attention:   {', '.join(attn)}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Epochs:      {epochs}")

    results = tma.train_attention_variants(config)
    tma.print_comparison_table(results)
    tma.write_comparison_artifacts(results, output_dir)
    volume.commit()
    return [res.to_row() for res in results]


class ModelManager:
    """Lazy loader for trained attention models and shared preprocessing assets."""

    _core_module = None
    _feature_extractor = None
    _tokenizer = None
    _index_word = None
    _models = {}

    @classmethod
    def _core(cls):
        if cls._core_module is None:
            import sys

            if "/root" not in sys.path:
                sys.path.insert(0, "/root")
            import caption_core as core  # pylint: disable=import-error

            cls._core_module = core
        return cls._core_module

    @classmethod
    def _ensure_tokenizer(cls):
        if cls._tokenizer is None or cls._index_word is None:
            tokenizer_path = DATASET_DIR / "tokenizer.pkl"
            if not tokenizer_path.exists():
                raise FileNotFoundError("Tokenizer file not found in dataset volume.")
            tokenizer, index_word = cls._core().load_tokenizer(str(tokenizer_path))
            cls._tokenizer = tokenizer
            cls._index_word = index_word
        return cls._tokenizer, cls._index_word

    @classmethod
    def _ensure_feature_extractor(cls):
        if cls._feature_extractor is None:
            cls._feature_extractor = cls._core().load_feature_extractor()
        return cls._feature_extractor

    @classmethod
    def _load_model(cls, attention: str):
        if attention in cls._models:
            return cls._models[attention]
        model_path = OUTPUT_DIR / f"caption_{attention}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for attention '{attention}'.")
        model, max_len = cls._core().load_caption_model(str(model_path))
        cls._models[attention] = (model, max_len)
        return cls._models[attention]

    @classmethod
    def generate_caption(cls, attention: str, image_bytes: bytes) -> str:
        attention = attention.lower()
        if not attention:
            raise ValueError("attention type is required")

        model, max_len = cls._load_model(attention)
        tokenizer, index_word = cls._ensure_tokenizer()

        processed = cls._core().process_image_from_bytes(image_bytes)
        features = cls._ensure_feature_extractor().predict(processed)
        caption = cls._core().predict_caption(model, features, tokenizer, index_word, max_len)
        return caption


def _available_models() -> List[dict]:
    models = []
    for attention in SUPPORTED_ATTENTION_TYPES:
        model_path = OUTPUT_DIR / f"caption_{attention}.keras"
        if model_path.exists():
            models.append({"attention": attention, "path": str(model_path)})
    if not models:
        for path in sorted(OUTPUT_DIR.glob("caption_*.keras")):
            attention = path.stem.replace("caption_", "")
            models.append({"attention": attention, "path": str(path)})
    return models


@app.function(image=TRAIN_IMAGE, volumes={"/data": volume}, memory=16384, timeout=300)
def list_available_models() -> List[dict]:
    """Return metadata for trained models available in the Modal volume."""
    models = _available_models()
    if not models:
        raise FileNotFoundError("No trained models found in /data/models. Run training first.")
    return models


@app.function(image=TRAIN_IMAGE, volumes={"/data": volume}, memory=16384, timeout=600)
def generate_caption_remote(attention_type: str, image_bytes: bytes) -> str:
    """Generate a caption for an uploaded image using the selected attention model."""
    models = _available_models()
    valid = {m["attention"] for m in models}
    if attention_type not in valid:
        raise FileNotFoundError(
            f"Model '{attention_type}' is not trained. Available: {', '.join(sorted(valid)) or 'none'}"
        )
    return ModelManager.generate_caption(attention_type, image_bytes)


def _build_fastapi_app() -> FastAPI:
    if FastAPI is None or UploadFile is None or File is None or HTTPException is None or CORSMiddleware is None:
        raise RuntimeError(
            "FastAPI dependencies are unavailable. Ensure fastapi is installed when running the API locally."
        )

    api = FastAPI(title="Image Caption API", version="1.0.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/")
    async def root():
        return {
            "status": "ok",
            "message": "Image captioning API running on Modal.",
            "endpoints": {
                "/models": "GET - list trained attention models",
                "/generate-caption": "POST - upload image + attention_type form field",
            },
        }

    @api.get("/models")
    async def get_models():
        try:
            models = list_available_models.remote()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"models": models}

    @api.post("/generate-caption")
    async def generate_caption_endpoint(
        attention_type: str,
        file: UploadFile = File(...),
    ):
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        try:
            caption = generate_caption_remote.remote(attention_type.lower(), data)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(status_code=500, detail=f"Failed to generate caption: {exc}") from exc
        return {
            "attention_type": attention_type.lower(),
            "caption": caption,
            "status": "success",
        }

    return api


@app.function(image=TRAIN_IMAGE, volumes={"/data": volume}, memory=8192, timeout=600)
@modal.asgi_app()
def fastapi_app():
    """Expose FastAPI service so Lovable can upload images and request captions."""
    return _build_fastapi_app()


@app.local_entrypoint()
def main(
    attention: str = "all",
    batch_size: int = 256,
    epochs: int = 20,
    learning_rate: float = 1e-3,
):
    """Convenience wrapper: modal run deploy.py --attention luong,bahdanau --epochs 10"""

    if attention.lower() == "all":
        attn = None
    else:
        attn = [part.strip() for part in attention.split(",") if part.strip()]

    summary = run_training.remote(
        attention_types=attn,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    print("Training run complete. Summary:")
    for item in summary:
        print(f"- {item['attention']}: best_val_loss={item['best_val_loss']:.4f}")
