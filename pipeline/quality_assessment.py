"""
Post-Training Quality Assessment

Evaluates Gaussian Splat training quality by rendering test views
and computing PSNR/SSIM metrics against ground truth frames.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from rich.console import Console

console = Console()


class QualityError(Exception):
    """Error during quality assessment."""
    pass


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images (grayscale)."""
    import cv2
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def split_train_test_frames(
    transforms_path: Path,
    test_every: int = 10,
) -> Tuple[Path, Path]:
    """
    Split transforms.json into train and test sets.

    Takes every Nth frame as test, remainder as train.
    Writes train_transforms.json and test_transforms.json.

    Returns:
        Tuple of (train_transforms_path, test_transforms_path)
    """
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)

    all_frames = transforms['frames']
    train_frames = []
    test_frames = []

    for i, frame in enumerate(all_frames):
        if i % test_every == 0:
            test_frames.append(frame)
        else:
            train_frames.append(frame)

    # Write train set
    train_transforms = {**transforms, 'frames': train_frames}
    train_path = transforms_path.parent / "train_transforms.json"
    with open(train_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)

    # Write test set
    test_transforms = {**transforms, 'frames': test_frames}
    test_path = transforms_path.parent / "test_transforms.json"
    with open(test_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)

    console.print(f"[blue]Split frames: {len(train_frames)} train, "
                  f"{len(test_frames)} test (every {test_every}th)[/blue]")

    return train_path, test_path


def evaluate_model(
    model_dir: Path,
    test_transforms_path: Path,
    output_dir: Path,
) -> Dict:
    """
    Evaluate a trained model against test views using ns-eval.

    Falls back to manual rendering + metric computation if ns-eval
    is not available or fails.

    Returns:
        Dict with PSNR, SSIM, and quality verdict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try ns-eval first
    config_path = model_dir / "config.yml"
    if config_path.exists():
        try:
            cmd = [
                'ns-eval',
                '--load-config', str(config_path),
                '--output-path', str(output_dir / "eval_results.json"),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and (output_dir / "eval_results.json").exists():
                with open(output_dir / "eval_results.json") as f:
                    eval_data = json.load(f)
                psnr = eval_data.get('results', {}).get('psnr', 0)
                ssim = eval_data.get('results', {}).get('ssim', 0)
                return _make_verdict(psnr, ssim, source="ns-eval")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    console.print("[yellow]ns-eval not available, using manual evaluation[/yellow]")

    # Manual evaluation: render test views and compare
    return _evaluate_manual(model_dir, test_transforms_path, output_dir)


def _evaluate_manual(
    model_dir: Path,
    test_transforms_path: Path,
    output_dir: Path,
) -> Dict:
    """Manual evaluation by rendering and comparing frames."""
    import cv2

    # Try to render test views using ns-render
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    config_path = model_dir / "config.yml"
    if not config_path.exists():
        console.print("[yellow]No config.yml found, skipping quality evaluation[/yellow]")
        return _make_verdict(0, 0, source="skipped")

    try:
        cmd = [
            'ns-render', 'dataset',
            '--load-config', str(config_path),
            '--data', str(test_transforms_path.parent),
            '--output-path', str(renders_dir),
            '--split', 'test',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            console.print(f"[yellow]ns-render failed, skipping metrics[/yellow]")
            return _make_verdict(0, 0, source="render_failed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        console.print("[yellow]ns-render not available, skipping metrics[/yellow]")
        return _make_verdict(0, 0, source="render_unavailable")

    # Load test transforms to find ground truth paths
    with open(test_transforms_path) as f:
        test_data = json.load(f)

    psnr_values = []
    ssim_values = []

    rendered_files = sorted(renders_dir.glob("*.png")) + sorted(renders_dir.glob("*.jpg"))

    for i, frame in enumerate(test_data['frames']):
        gt_path = test_transforms_path.parent / frame['file_path']
        if not gt_path.exists() or i >= len(rendered_files):
            continue

        gt_img = cv2.imread(str(gt_path))
        render_img = cv2.imread(str(rendered_files[i]))

        if gt_img is None or render_img is None:
            continue

        # Resize if dimensions differ
        if gt_img.shape[:2] != render_img.shape[:2]:
            render_img = cv2.resize(render_img, (gt_img.shape[1], gt_img.shape[0]))

        psnr_values.append(compute_psnr(gt_img, render_img))
        ssim_values.append(compute_ssim(gt_img, render_img))

    if not psnr_values:
        console.print("[yellow]No valid comparison pairs found[/yellow]")
        return _make_verdict(0, 0, source="no_pairs")

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    return _make_verdict(mean_psnr, mean_ssim, source="manual")


def _make_verdict(psnr: float, ssim: float, source: str = "") -> Dict:
    """Create quality verdict from metrics."""
    if source in ("skipped", "render_failed", "render_unavailable", "no_pairs"):
        verdict = "unknown"
        console.print(f"[yellow]Quality assessment: {source} (no metrics available)[/yellow]")
    elif psnr < 20:
        verdict = "fail"
        console.print(f"[bold red]Quality: FAIL — PSNR={psnr:.1f} dB, SSIM={ssim:.3f}[/bold red]")
        console.print("[red]Training result is likely unusable.[/red]")
    elif psnr < 25:
        verdict = "marginal"
        console.print(f"[yellow]Quality: MARGINAL — PSNR={psnr:.1f} dB, SSIM={ssim:.3f}[/yellow]")
        console.print("[yellow]Result may have visible artifacts.[/yellow]")
    else:
        verdict = "pass"
        console.print(f"[green]Quality: PASS — PSNR={psnr:.1f} dB, SSIM={ssim:.3f}[/green]")

    return {
        "psnr": psnr,
        "ssim": ssim,
        "verdict": verdict,
        "source": source,
    }
