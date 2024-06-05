from PIL import Image, ImageDraw, ImageFont
import os
import random
import string
import numpy as np
from tqdm import tqdm

def generate_random_text(max_length=100):
    """Generates a random string of varying length up to max_length, including spaces, commas, and periods with spaces more frequently."""
    characters = string.ascii_letters + string.digits + ' ,.'  # Existing characters
    characters += ' ' * 10  # Adding additional spaces to increase their likelihood
    length = random.randint(10, max_length)  # Random length between 10 and max_length
    return ''.join(random.choice(characters) for _ in range(length))

def add_edge_shadows(draw, image_size):
    """Add deep black shadows in the form of rectangles along the edges."""
    width, height = image_size
    edge_thickness = random.randint(20, 50)  # Adjust the thickness of the edge shadow
    draw.rectangle([0, 0, width, edge_thickness], fill='black')  # Top shadow
    draw.rectangle([0, height - edge_thickness, width, height], fill='black')  # Bottom shadow

def generate_images_from_fonts(num_images=300, base_output_dir='data'):
    fonts_folder = 'fonts'
    font_files = [os.path.join(fonts_folder, f) for f in os.listdir(fonts_folder) if f.endswith('.ttf')]

    for font_path in tqdm(font_files):
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        output_dir = os.path.join(base_output_dir, font_name)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(num_images):
            image_width = random.randint(200, 1200)
            image_height = random.randint(200, 500)
            image_size = (image_width, image_height)

            text = generate_random_text()
            image = Image.new('RGB', image_size, 'white')
            draw = ImageDraw.Draw(image)

            # Start with a large font size and reduce if necessary
            fontsize = int(image_size[1] * 0.5)  # Start with 50% of image height
            font = ImageFont.truetype(font_path, size=fontsize)

            # Adjust font size until the text fits the image
            while True:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                if text_width > image_size[0] or text_height > image_size[1]:
                    fontsize -= 1
                    font = ImageFont.truetype(font_path, size=fontsize)
                    if fontsize <= 10:  # Stop reducing size at a reasonable minimum
                        break
                else:
                    break

            # Draw the text centered on the image
            x = (image_width - text_width) // 2
            y = (image_height - text_height) // 2
            draw.text((x, y), text, font=font, fill='black')

            # Add shadows after drawing text
            add_edge_shadows(draw, image_size)

            image_file = os.path.join(output_dir, f"{text}_{i}.png")
            image.save(image_file)

# Example usage
generate_images_from_fonts()
