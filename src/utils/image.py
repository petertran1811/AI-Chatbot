from PIL import Image
from constants import IMAGE_CHARACTERS_DIR


def get_image(name: str) -> Image:
    image_path = f"{IMAGE_CHARACTERS_DIR}/{name}.jpg"
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(e)
        image = f"{IMAGE_CHARACTERS_DIR}/Ava.jpg"

    return image
