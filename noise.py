import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def load_image(image_path: str) -> np.array:
    img = Image.open(image_path)
    return np.array(img).astype(np.float32) / 255.0  # Normalizacja do [0, 1]


def add_gaussian_noise(
        image: np.array,
        mean: float = 0.0,
        std: float = 0.1
        ) -> np.array:
    """
    Dodaje gaussowski szum do obrazu.

    :param image: Obraz wejściowy jako tablica NumPy o wartościach z [0, 1].
    :param mean: Średnia gaussowskiego szumu.
    :param std: Odchylenie standardowe gaussowskiego szumu.
    :return: Obraz z dodanym szumem, przeskalowany do zakresu [0, 1].
    """
    noise = np.random.normal(mean, std, image.shape)

    noisy_image = image + noise

    noisy_image_clipped = np.clip(noisy_image, 0, 1)

    return noisy_image_clipped


def get_noised_images(image, n: int = 10, noise: float = 0.5) -> list:
    x_list = [image]
    for _ in range(n-1):
        new_x = add_gaussian_noise(x_list[-1], std=noise)
        x_list.append(new_x)

    return x_list


def draw_noise_evolution(
        image,
        n: int = 10,
        noise: float = 0.5,
        save_path: str = None
        ) -> None:
    images = get_noised_images(image, n, noise)
    plt.figure(figsize=(n*5+5, 5))

    plt.subplot(1, n, 1)
    plt.imshow(images[0], cmap="gray")
    plt.title(f"Noisy {0}")
    for i, image in enumerate(images[0:]):
        plt.subplot(1, n, i+1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Noisy {i}")

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    image_path = "images/ThinkPad_1024.png"
    image = load_image(image_path)

    draw_noise_evolution(image, 4, 1.6, "images/noise.png")
