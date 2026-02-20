import os
import cv2
import numpy as np
from tqdm import tqdm

# === CONFIG ===
RAW_DIR = "data/signatures/raw/genuine"
PROCESSED_DIR = "data/signatures/processed"
IMG_SIZE = 64  # As per project specification

os.makedirs(PROCESSED_DIR, exist_ok=True)


def crop_signature(img):
    """
    Crop excess white margins around the signature
    """
    # Convert to binary (threshold)
    _, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)

    # Find non-zero (ink) coordinates
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img[y:y+h, x:x+w]
        return cropped
    else:
        return img  # fallback if blank


def preprocess_image(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    # Crop margins (remove whitespace)
    img = crop_signature(img)

    # Resize to 64x64 (spec requirement)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize to [-1, 1] for GAN (tanh output compatibility)
    img = img.astype(np.float32) / 127.5 - 1.0

    return img


def main():
    image_files = [
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Found {len(image_files)} raw signature images.")

    processed_count = 0

    for filename in tqdm(image_files, desc="Preprocessing signatures"):
        input_path = os.path.join(RAW_DIR, filename)
        output_path = os.path.join(PROCESSED_DIR, filename)

        processed_img = preprocess_image(input_path)

        if processed_img is not None:
            # Convert back to [0,255] for saving visualization
            save_img = ((processed_img + 1) * 127.5).astype(np.uint8)
            cv2.imwrite(output_path, save_img)
            processed_count += 1

    print(f"\n✅ Preprocessing complete!")
    print(f"Processed images saved to: {PROCESSED_DIR}")
    print(f"Total processed: {processed_count}")


if __name__ == "__main__":
    main()