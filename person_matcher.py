#!/usr/bin/env python3
"""Command-line program to compare two face images and determine if they are the same person."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

# Suppress harmless urllib3 LibreSSL warnings on macOS
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# ArcFace is substantially more tolerant of normal appearance changes (facial
# hair, age, lighting, and camera angle) than Facenet512 for this application.
# Its DeepFace-provided threshold remains model-specific and is not shared with
# the other recognition models.
DEFAULT_MODEL = "ArcFace"
DEFAULT_DETECTOR = "opencv"
DEFAULT_METRIC = "cosine"


def clean_path(path_str: str) -> Path:
    """Strip quotes and whitespace from drag-and-dropped terminal paths."""
    cleaned = path_str.strip().strip("'\"")
    return Path(cleaned).expanduser().resolve()


def validate_image_file(path: Path, label: str) -> bool:
    """Check existence, read permissions, and image decode integrity."""
    if not path.exists():
        print(f"Error: {label} image does not exist: {path}", file=sys.stderr)
        return False
    if not path.is_file():
        print(f"Error: {label} path is not a file: {path}", file=sys.stderr)
        return False

    # Check file read permissions (common macOS permission restriction on ~/Downloads, ~/Desktop)
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            if not chunk:
                print(f"Error: {label} image file is empty (0 bytes): {path}", file=sys.stderr)
                return False
    except PermissionError:
        print(
            f"\nError: Permission denied reading {label.lower()} image:\n  {path}\n\n"
            f"macOS blocks terminal access to ~/Downloads, ~/Desktop, and ~/Documents by default.\n\n"
            f"Quick fix:\n"
            f"  Copy or move the photos into this project folder and run again:\n"
            f"    cp \"{path}\" ./{label.lower()}.jpg\n"
            f"    python3 person_matcher.py {label.lower()}.jpg <other_image>\n\n"
            f"Alternative fix:\n"
            f"  Open macOS System Settings → Privacy & Security → Files and Folders,\n"
            f"  and allow your Terminal or IDE to access the Downloads folder.",
            file=sys.stderr,
        )
        return False
    except Exception as exc:
        print(f"Error: Cannot read {label.lower()} image ({exc}): {path}", file=sys.stderr)
        return False

    # Verify OpenCV can decode the image
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            print(
                f"\nError: Could not decode {label.lower()} image:\n  {path}\n"
                f"Please ensure the file is a valid, uncorrupted image (JPEG, PNG, WEBP, etc.).",
                file=sys.stderr,
            )
            return False
    except Exception:
        pass

    return True


def verify_faces(
    image1_path: Path,
    image2_path: Path,
    model_name: str = DEFAULT_MODEL,
    detector: str = DEFAULT_DETECTOR,
    metric: str = DEFAULT_METRIC,
) -> dict:
    """Compare two images and return the DeepFace verification result."""
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise RuntimeError(
            "DeepFace is not installed. Run: python3 -m pip install -r requirements.txt"
        ) from exc

    return DeepFace.verify(
        img1_path=str(image1_path),
        img2_path=str(image2_path),
        model_name=model_name,
        detector_backend=detector,
        distance_metric=metric,
        enforce_detection=True,
        align=True,
        silent=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two face photos to determine if they are the same person.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image1",
        nargs="?",
        help="Path to the first image file (prompted interactively if omitted)",
    )
    parser.add_argument(
        "image2",
        nargs="?",
        help="Path to the second image file (prompted interactively if omitted)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Face recognition model (e.g. Facenet512, VGG-Face, ArcFace, GhostFaceNet)",
    )
    parser.add_argument(
        "--detector",
        default=DEFAULT_DETECTOR,
        help="Face detector backend (e.g. opencv, retinaface, mtcnn, ssd)",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=["cosine", "euclidean", "euclidean_l2"],
        help="Distance metric used for comparison",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Prompt interactively if paths are not supplied via CLI arguments
    img1_input = args.image1
    if not img1_input:
        try:
            img1_input = input("Enter path to first image: ")
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return 130

    img2_input = args.image2
    if not img2_input:
        try:
            img2_input = input("Enter path to second image: ")
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return 130

    img1_path = clean_path(img1_input)
    img2_path = clean_path(img2_input)

    # Validate image file existence and permissions
    if not validate_image_file(img1_path, "First"):
        return 1
    if not validate_image_file(img2_path, "Second"):
        return 1

    print(f"\nAnalyzing images...")
    print(f"• Image 1: {img1_path.name}")
    print(f"• Image 2: {img2_path.name}")
    print(f"• Model:   {args.model} ({args.metric} distance)\n")

    try:
        result = verify_faces(
            image1_path=img1_path,
            image2_path=img2_path,
            model_name=args.model,
            detector=args.detector,
            metric=args.metric,
        )
    except ValueError as exc:
        msg = str(exc)
        if "Face could not be detected" in msg:
            print("Error: Could not detect a face in one of the images.", file=sys.stderr)
            print("Please ensure both photos contain a clear, front-facing face.", file=sys.stderr)
        else:
            print(f"Error during face comparison: {msg}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error during face comparison: {exc}", file=sys.stderr)
        return 1

    verified = bool(result.get("verified", False))
    distance = float(result.get("distance", 0.0))
    threshold = float(result.get("threshold", 0.0))

    separator = "=" * 50
    print(separator)
    if verified:
        print("VERDICT: MATCH (Likely the same person)")
    else:
        print("VERDICT: NO MATCH (Different persons)")
    print(separator)
    print(f"Distance:  {distance:.4f}")
    print(f"Threshold: {threshold:.4f} (lower distance means higher similarity)")
    print(f"Verified:  {verified}")
    print(separator)

    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
