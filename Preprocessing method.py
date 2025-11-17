import cv2
import os
from google.colab.patches import cv2_imshow

def display_and_save_uniform_clips(main_folder, output_folder, T=16):
    """
    Extracts T-frame clips from each video in the main_folder and saves them
    in the output_folder, preserving subfolder structure.

    Args:
        main_folder (str): Path to the folder containing videos.
        output_folder (str): Path to save the extracted frames.
        T (int): Number of frames to sample per video.
    """
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith(('.avi', '.mp4')):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, main_folder)
                save_subfolder = os.path.join(output_folder, relative_path, os.path.splitext(file)[0])
                os.makedirs(save_subfolder, exist_ok=True)

                print(f"\n📽️ Processing: {video_path}")

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"❌ Could not open {video_path}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < T:
                    print(f"⚠️ Video too short ({total_frames} frames). Skipping.")
                    cap.release()
                    continue

                # Compute frame stride Δ
                delta = total_frames // T
                print(f"Total frames: {total_frames}, Sampling stride Δ: {delta}")

                for i in range(T):
                    frame_idx = i * delta
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"⚠️ Failed to read frame {frame_idx}")
                        break

                    # Display frame (optional)
                    cv2_imshow(frame)
                    print(f"🖼️ Displayed Frame {frame_idx}")

                    # Save frame
                    output_path = os.path.join(save_subfolder, f'frame_{i:03}.jpg')
                    cv2.imwrite(output_path, frame)
                    print(f"💾 Saved Frame {i} to {output_path}")

                cap.release()
    print("\n✅ All videos processed.")

# ==========================
# Example usage
# ==========================
main_folder_path = '/content/drive/MyDrive/Colab Notebooks/DCSASS/DCSASS Dataset'
output_folder_path = '//content/drive/MyDrive/Colab Notebooks/DCSASS/Extracted_Frames'

# Extract T=16 frames per video
display_and_save_uniform_clips(main_folder_path, output_folder_path, T=16)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------
# Configuration
# ------------------------------
target_size = (224, 224)
mu = np.array([0.485, 0.456, 0.406])
sigma = np.array([0.229, 0.224, 0.225])

input_folder = '//content/drive/MyDrive/Colab Notebooks/DCSASS/Extracted_Frames'
output_folder = '//content/drive/MyDrive/Colab Notebooks/DCSASS/Processed_Frames'
os.makedirs(output_folder, exist_ok=True)

# ------------------------------
# Functions
# ------------------------------

def normalize_image(img):
    """Normalize image per channel using mu and sigma."""
    img_norm = img / 255.0
    img_norm = (img_norm - mu) / sigma
    return img_norm

def resize_image(img, size=target_size):
    """Resize and convert to RGB."""
    resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb_image

def display_images(input_image, output_image, title=""):
    """Display original vs processed image."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(output_image)
    axs[1].set_title("Processed Image")
    axs[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# ------------------------------
# Process all images
# ------------------------------
for root, dirs, files in os.walk(input_folder):
    for file in tqdm(files):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error reading {file}")
                continue

            # Resize
            img_resized = resize_image(img)

            # Normalize
            img_normalized = normalize_image(img_resized)

            # Convert back to displayable format for visualization
            img_display = ((img_normalized * sigma + mu) * 255.0).clip(0, 255).astype(np.uint8)

            # Save processed image (optional)
            relative_path = os.path.relpath(root, input_folder)
            save_dir = os.path.join(output_folder, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file)
            cv2.imwrite(save_path, cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))

            # Display
            display_images(img, img_display, title=file)
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------------
# Compute optical flow (Farneback)
# ------------------------------
def compute_motion(f1, f2):
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Convert flow to RGB image
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb

# ------------------------------
# Display two images side by side
# ------------------------------
def display_input_output(input_img, output_img, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    axs[0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Input Frame")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Saved Motion Field")
    axs[1].axis("off")

    plt.suptitle(title)
    plt.show()
    plt.close()

# ------------------------------
# Main input and output folders
# ------------------------------
main_input_folder = "/content/drive/MyDrive/Colab Notebooks/DCSASS/Processed_Frames"
main_output_folder = "/content/drive/MyDrive/Colab Notebooks/DCSASS/Processed_Frames/output_folder"
os.makedirs(main_output_folder, exist_ok=True)

# ------------------------------
# Process all subfolders
# ------------------------------
for root, dirs, files in os.walk(main_input_folder):
    image_files = sorted([f for f in files if f.lower().endswith(('.png', '.jpg'))])

    if len(image_files) < 2:
        continue  # Need at least 2 images to compute motion

    rel_path = os.path.relpath(root, main_input_folder)
    output_folder = os.path.join(main_output_folder, rel_path)
    os.makedirs(output_folder, exist_ok=True)

    image_paths = [os.path.join(root, f) for f in image_files]

    for i in tqdm(range(len(image_paths)-1), desc=f"Processing {rel_path}"):
        f1 = cv2.imread(image_paths[i])
        f2 = cv2.imread(image_paths[i+1])

        # Compute motion field
        motion = compute_motion(f1, f2)

        # Save input frame
        input_filename = f"{os.path.splitext(image_files[i])[0]}_input.png"
        cv2.imwrite(os.path.join(output_folder, input_filename), f1)

        # Save motion field
        motion_filename = f"{os.path.splitext(image_files[i])[0]}_motion.png"
        motion_path = os.path.join(output_folder, motion_filename)
        cv2.imwrite(motion_path, motion)

        # Reload saved motion field to display
        saved_motion = cv2.imread(motion_path)

        # Display input and saved motion field
        display_input_output(f1, saved_motion, title=image_files[i])
import cv2
import os
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
# ---------------------------
crop_size = (224, 224)
noise_std = 0.05
brightness_jitter = 0.2
contrast_jitter = 0.2
img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# ---------------------------
# Input / Output
# ---------------------------
input_main = "/content/drive/MyDrive/Colab Notebooks/DCSASS/Processed_Frames/output_folder"
output_main = "/content/drive/MyDrive/Colab Notebooks/DCSASS/Augmented_Display"
os.makedirs(output_main, exist_ok=True)

# ---------------------------
# Augmentation functions
# ---------------------------
def random_crop(frame):
    transform = T.Compose([T.ToPILImage(), T.RandomCrop(crop_size), T.ToTensor()])
    f_aug = transform(frame)
    return (f_aug.permute(1,2,0).numpy()*255).astype(np.uint8), "RandomCrop"

def horizontal_flip(frame):
    transform = T.Compose([T.ToPILImage(), T.RandomHorizontalFlip(1.0), T.ToTensor()])
    f_aug = transform(frame)
    return (f_aug.permute(1,2,0).numpy()*255).astype(np.uint8), "Flip"

def brightness_jitter_func(frame):
    transform = T.Compose([T.ToPILImage(), T.ColorJitter(brightness=brightness_jitter), T.ToTensor()])
    f_aug = transform(frame)
    return (f_aug.permute(1,2,0).numpy()*255).astype(np.uint8), "Brightness"

def contrast_jitter_func(frame):
    transform = T.Compose([T.ToPILImage(), T.ColorJitter(contrast=contrast_jitter), T.ToTensor()])
    f_aug = transform(frame)
    return (f_aug.permute(1,2,0).numpy()*255).astype(np.uint8), "Contrast"

def gaussian_noise(frame):
    noise = np.random.normal(0, noise_std, frame.shape).astype(np.float32)
    f_aug = np.clip(frame.astype(np.float32)/255.0 + noise, 0, 1) * 255
    return f_aug.astype(np.uint8), "Noise"

# ---------------------------
# Apply all augmentations
# ---------------------------
def augment_frame(frame):
    techniques = [
        random_crop,
        horizontal_flip,
        brightness_jitter_func,
        contrast_jitter_func,
        gaussian_noise
    ]
    augmented = []
    for func in techniques:
        out, name = func(frame)
        augmented.append((out, name))
    return augmented

# ---------------------------
# Load folder images
# ---------------------------
def load_images(main_folder):
    all_imgs = []
    folder_paths = []

    for root, dirs, files in os.walk(main_folder):
        imgs = [os.path.join(root, f) for f in files if f.lower().endswith(img_extensions)]
        if imgs:
            all_imgs.append(sorted(imgs))
            folder_paths.append(root)

    return all_imgs, folder_paths

# ---------------------------
# Load all image sequences
# ---------------------------
all_sequences, folder_paths = load_images(input_main)

# ---------------------------
# Process each folder
# ---------------------------
for seq_idx, images in enumerate(all_sequences):

    frames = [cv2.imread(p) for p in images]
    filenames = [os.path.basename(p) for p in images]

    # Create corresponding output folder
    relative_path = os.path.relpath(folder_paths[seq_idx], input_main)
    save_folder = os.path.join(output_main, relative_path)
    os.makedirs(save_folder, exist_ok=True)

    for i, frame in enumerate(frames):

        # Augment
        aug_list = augment_frame(frame)

        # Prepare horizontal display list
        disp_frames = [frame]
        titles = ["Original"]

        for img_aug, tname in aug_list:
            disp_frames.append(img_aug)
            titles.append(tname)

        # Resize all to same height
        th = max(f.shape[0] for f in disp_frames)
        resized = []
        for f in disp_frames:
            h, w = f.shape[:2]
            scale = th / h
            rw = int(w * scale)
            resized.append(cv2.resize(f, (rw, th)))

        # Merge horizontally
        merged = np.hstack(resized)

        # Add label strip
        label_h = 35
        final_img = np.ones((th + label_h, merged.shape[1], 3), dtype=np.uint8) * 255
        final_img[label_h:, :, :] = merged

        # Draw labels
        x = 0
        for f, text in zip(resized, titles):
            w = f.shape[1]
            cv2.putText(final_img, text, (x + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            x += w

        # Display
        plt.figure(figsize=(20, 6))
        plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # Save **folder-wise**
        save_name = os.path.splitext(filenames[i])[0] + "_augmented.png"
        cv2.imwrite(os.path.join(save_folder, save_name), final_img)

        # Save each augmented separately also
        for aug_img, tname in aug_list:
            sep_name = os.path.splitext(filenames[i])[0] + f"_{tname}.png"
            cv2.imwrite(os.path.join(save_folder, sep_name), aug_img)

