import os
import random
from PIL import Image


def overlay_images(subject_path, background_path, output_path):
    subjects = os.listdir(subject_path)
    backgrounds = os.listdir(background_path)

    subject_image = Image.open(os.path.join(subject_path, random.choice(subjects)))

    # Choose a random background image
    background_image_path = os.path.join(background_path, random.choice(backgrounds))
    background_image = Image.open(background_image_path)

    # Resize subject image to fit within the background
    max_width = min(subject_image.width, background_image.width)
    max_height = min(subject_image.height, background_image.height)
    subject_image = subject_image.resize((max_width, max_height))

    # Choose a random position for overlay
    x = random.randint(0, background_image.width - subject_image.width)
    y = random.randint(0, background_image.height - subject_image.height)

    # Create a transparent overlay image of the subject
    overlay = Image.new("RGBA", background_image.size)
    overlay.paste(subject_image, (x, y))

    # Blend the overlay onto the background
    overlaid_image = Image.alpha_composite(background_image.convert("RGBA"), overlay)

    # Save the overlaid image
    overlaid_image.save(output_path)
    print(f"Overlaid image saved at {output_path}")


if __name__ == "__main__":
    subjects_folder = "subjects"
    backgrounds_folder = "backgrounds"
    output_folder = "output"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    overlay_images(subjects_folder, backgrounds_folder, os.path.join(output_folder, "overlaid_image.png"))
