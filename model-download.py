import argparse
import os
import sys
from typing import Iterable, List

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def _as_set(items: Iterable[str]) -> set:
    return {item.strip().lower() for item in items if item and item.strip()}


def _download_snapshot(repo_id: str, cache_dir: str) -> None:
    print(f"[download] {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        resume_download=True,
        local_files_only=False,
    )


# DenseNet手动拉取（支持断点续传）：wget -c --tries=10 --timeout=30 https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt -O /root/.torchxrayvision/models_data/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt

def _download_xrv_models() -> None:
    # torchxrayvision uses its own cache directory; this triggers weight downloads.
    import torchxrayvision as xrv

    print("[download] torchxrayvision DenseNet weights")
    _ = xrv.models.DenseNet(weights="densenet121-res224-all")
    print("[download] torchxrayvision PSPNet weights")
    _ = xrv.baseline_models.chestx_det.PSPNet()

def _try_download_xrv_models() -> None:
    try:
        _download_xrv_models()
    except Exception as exc:
        # Avoid aborting the rest of the downloads if XRV fails.
        print(f"[warn] torchxrayvision download failed: {exc}")
        print(
            "[warn] You can retry later or download manually, e.g.:\n"
            "  wget -c --tries=10 --timeout=30 "
            "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
            "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-"
            "tr15-sc15-seed0-best.pt "
            "-O /root/.torchxrayvision/models_data/"
            "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-"
            "tr15-sc15-seed0-best.pt"
        )

def download_models(model_dir: str, tools: List[str], include_xrv: bool) -> None:
    tools_set = _as_set(tools)
    if "all" in tools_set or not tools_set:
        tools_set = {
            "classifier",
            "segmentation",
            "report",
            "vqa",
            "grounding",
            "llava",
        }

    if include_xrv and {"classifier", "segmentation"} & tools_set:
        _try_download_xrv_models()

    if "report" in tools_set:
        _download_snapshot("IAMJB/chexpert-mimic-cxr-findings-baseline", model_dir)
        _download_snapshot("IAMJB/chexpert-mimic-cxr-impression-baseline", model_dir)

    if "vqa" in tools_set:
        _download_snapshot("StanfordAIMI/CheXagent-2-3b", model_dir)

    if "grounding" in tools_set:
        _download_snapshot("microsoft/maira-2", model_dir)

    if "llava" in tools_set:
        _download_snapshot("microsoft/llava-med-v1.5-mistral-7b", model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MedRAX model weights without requiring a GPU."
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MEDRAX_MODEL_DIR", "/data/huggingface_home"),
        help="Directory for Hugging Face cache (default: /data/huggingface_home).",
    )
    parser.add_argument(
        "--tools",
        default="all",
        help=(
            "Comma-separated list: classifier,segmentation,report,vqa,grounding,llava "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--no-xrv",
        action="store_true",
        help="Skip torchxrayvision weight downloads.",
    )
    return parser.parse_args()


def main() -> int:
    # Load .env from current working directory if present.
    load_dotenv()
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    args = parse_args()
    tools = [t for t in args.tools.split(",") if t.strip()]
    model_dir = os.path.abspath(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)

    try:
        download_models(model_dir, tools, include_xrv=not args.no_xrv)
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    print("[done] all requested models downloaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())

