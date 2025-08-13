from torch import Tensor
import einops
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch as t


def download_dataset():
    """Downloads the CelebA dataset and saves it to the specified directory."""
    os.makedirs(celeb_image_dir, exist_ok=True)

    celeb_data_dir = "data/celeba"
    celeb_image_dir = celeb_data_dir / "img_align_celeba"

    if len(list(celeb_image_dir.glob("*.jpg"))) > 0:
        print("Dataset already loaded.")
    else:
        dataset = load_dataset("nielsr/CelebA-faces")
        print("Dataset loaded.")

        for idx, item in tqdm(
            enumerate(dataset["train"]), total=len(dataset["train"]), desc="Saving imgs...", ascii=True
        ):
            # The image is already a JpegImageFile, so we can directly save it
            item["image"].save(celeb_image_dir / f"{idx:06}.jpg")

        print("All images have been saved.")

def get_dataset(train: bool = True) -> Dataset:

    image_size = 128
    assert train, "CelebA dataset only has a training set"
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = datasets.ImageFolder(
        root="data/celeba", transform=transform
    )

    return trainset

def display_data(x: Tensor, nrows: int, title: str):
    """Displays a batch of data, using plotly."""
    ncols = x.shape[0] // nrows
    # Reshape into the right shape for plotting (make it 2D if image is monochrome)
    y = einops.rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows).squeeze()
    # Normalize in the 0-1 range, then map to integer type
    y = (y - y.min()) / (y.max() - y.min())
    y = (y * 255).to(dtype=t.uint8)
    # Display data
    plt.imshow(
        y,
        binary_string=(y.ndim == 2),
        height=100 * (nrows + 4),
        width=100 * (ncols + 5),
        title=f"{title}<br>single input shape = {x[0].shape}",
    )