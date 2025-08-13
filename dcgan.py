import os
from dataclasses import dataclass, field
from typing import Literal
import einops
import torch as t
import wandb
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import DataLoader
from jaxtyping import Float, Int
import numpy as np
from utils import get_dataset, display_data

from tqdm import tqdm


device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        sf = 1 / np.sqrt(in_features)

        weight = sf * (2 * t.rand(out_features, in_features) - 1)
        self.weight = nn.Parameter(weight)

        if bias:
            bias = sf * (2 * t.rand(out_features) - 1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        x = einops.einsum(x, self.weight, "... in_feats, out_feats in_feats -> ... out_feats")
        if self.bias is not None:
            x += self.bias
        return x

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(x, t.tensor(0.0))
    
class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)  # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)  # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: Tensor) -> Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            x = mod(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.

        We assume kernel is square, with height = width = `kernel_size`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_height = kernel_width = kernel_size
        sf = 1 / np.sqrt(in_channels * kernel_width * kernel_height)
        self.weight = nn.Parameter(sf * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])

IntOrPair = int | tuple[int, int]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

def pad2d(
    x: Float[Tensor, "batch in_channels height width"],
    left: int,
    right: int,
    top: int,
    bottom: int,
    pad_value: float,
) -> Float[Tensor, "batch in_channels height_padding width_padding"]:
    """Return a new tensor with padding applied to the width & height dimensions."""
    B, C, H, W = x.shape
    output = x.new_full(size=(B, C, top + H + bottom, left + W + right), fill_value=pad_value)
    output[..., top : top + H, left : left + W] = x
    return output

def fractional_stride_2d(
    x: Float[Tensor, "batch in_channels height width"], stride_h: int, stride_w: int
) -> Float[Tensor, "batch in_channels output_height output_width"]:
    """
    Same as fractional_stride_1d, except we apply it along the last 2 dims of x (height and width).
    """
    batch, in_channels, height, width = x.shape
    width_new = width + (stride_w - 1) * (width - 1)
    height_new = height + (stride_h - 1) * (height - 1)
    x_new_shape = (batch, in_channels, height_new, width_new)

    # Create an empty array to store the spaced version of x in.
    x_new = t.zeros(size=x_new_shape, dtype=x.dtype, device=x.device)

    x_new[..., ::stride_h, ::stride_w] = x

    return x_new

def conv2d_minimal(
    x: Float[Tensor, "batch in_channels height width"],
    weights: Float[Tensor, "out_channels in_channels kernel_height kernel_width"],
) -> Float[Tensor, "batch out_channels height_padding width_padding"]:
    """
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.
    """
    b, ic, h, w = x.shape
    oc, ic2, kh, kw = weights.shape
    assert ic == ic2, "in_channels for x and weights don't match up"
    ow = w - kw + 1
    oh = h - kh + 1

    s_b, s_ic, s_h, s_w = x.stride()

    # Get strided x (the new height/width dims have the same stride as the original height/width-strides of x)
    x_new_shape = (b, ic, oh, ow, kh, kw)
    x_new_stride = (s_b, s_ic, s_h, s_w, s_h, s_w)

    x_strided = x.as_strided(size=x_new_shape, stride=x_new_stride)

    return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")


def conv_transpose2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> Tensor:
    """Like torch's conv_transpose2d using bias=False
    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)
    Returns: shape (batch, out_channels, output_height, output_width)
    """
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)

    batch, ic, height, width = x.shape
    ic_2, oc, kernel_height, kernel_width = weights.shape
    assert ic == ic_2, f"in_channels for x and weights don't match up. Shapes are {x.shape}, {weights.shape}."

    # Apply spacing
    x_spaced_out = fractional_stride_2d(x, stride_h, stride_w)

    # Apply modification (which is controlled by the padding parameter)
    pad_h_actual = kernel_height - 1 - padding_h
    pad_w_actual = kernel_width - 1 - padding_w
    assert min(pad_h_actual, pad_w_actual) >= 0, "total amount padded should be positive"
    x_mod = pad2d(
        x_spaced_out, left=pad_w_actual, right=pad_w_actual, top=pad_h_actual, bottom=pad_h_actual, pad_value=0
    )

    # Modify weights
    weights_mod = einops.rearrange(weights.flip(-1, -2), "i o h w -> o i h w")

    # Return the convolution
    return conv2d_minimal(x_mod, weights_mod)
    
class ConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        """
        Same as torch.nn.ConvTranspose2d with bias=False.
        Name your weight field `self.weight` for compatibility with the tests.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = force_pair(kernel_size)
        self.stride = stride
        self.padding = padding

        sf = 1 / (self.out_channels * self.kernel_size[0] * self.kernel_size[1]) ** 0.5
        self.weight = nn.Parameter(sf * (2 * t.rand(in_channels, out_channels, *self.kernel_size) - 1))

    def forward(
        self, x: Float[Tensor, "batch in_channels height width"]
    ) -> Float[Tensor, "batch out_channels output_height output_width"]:
        return conv_transpose2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        # Calculating mean and var over all dims except for the channel dim
        if self.training:
            # Take mean over all dimensions except the feature dimension
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            # Updating running mean and variance, in line with PyTorch documentation
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        # Rearranging these so they can be broadcasted
        reshape = lambda x: einops.rearrange(x, "channels -> 1 channels 1 1")

        # Normalize, then apply affine transformation from self.weight & self.bias
        x_normed = (x - reshape(mean)) / (reshape(var) + self.eps).sqrt()
        x_affine = x_normed * reshape(self.weight) + reshape(self.bias)
        return x_affine

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])


class Tanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return (t.exp(x) - t.exp(-x))/(t.exp(x)+t.exp(-x))


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        max_part = t.maximum(x, t.tensor(0.0, device=x.device))
        min_part = t.minimum(x, t.tensor(0.0, device=x.device))
        return max_part + self.negative_slope * min_part
    

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class Sigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + t.exp(-x))
    
def initialize_weights(model: nn.Module) -> None:
    """
    Initializes weights according to the DCGAN paper (details at the end of page 3 of the DCGAN
    paper), by modifying the weights of the model in place.
    """
    for module in model.modules():
        if isinstance(module, (ConvTranspose2d, Conv2d, Linear)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        """
        Implements the generator architecture from the DCGAN paper (the diagram at the top
        of page 4). We assume the size of the activations doubles at each layer (so image
        size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            latent_dim_size:
                the size of the latent dimension, i.e. the input to the generator
            img_size:
                the size of the image, i.e. the output of the generator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the generator (starting closest
                to the middle of the DCGAN and going outward, i.e. in chronological order for
                the generator)
        """
        n_layers = len(hidden_channels)
        assert img_size % (2**n_layers) == 0, "activation size must double at each layer"

        super().__init__()
        hidden_channels = hidden_channels[::-1]
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        
        initial_height = img_size // (2**n_layers)
        initial_size = hidden_channels[0] * (initial_height**2)
        self.project_and_reshape = Sequential(
            Linear(latent_dim_size, initial_size, bias=False),
            Rearrange("b (c h w) -> b c h w", c=hidden_channels[0], h=initial_height, w=initial_height),
            BatchNorm2d(hidden_channels[0]),
            ReLU()
        )

        conv_layers = []
        in_channels = hidden_channels
        out_channels = hidden_channels[1:] + [img_channels] 
        for i, (ic, oc) in enumerate(zip(in_channels, out_channels)):
            if i < len(in_channels) - 1:
                conv_layers.extend([ConvTranspose2d(ic,oc,4,2,1), BatchNorm2d(oc), ReLU()])
            else:
                conv_layers.extend([ConvTranspose2d(ic,oc,4,2,1), Tanh()])

        self.hidden_layers = Sequential(*conv_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project_and_reshape(x)
        x = self.hidden_layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        """
        Implements the discriminator architecture from the DCGAN paper (the mirror image of
        the diagram at the top of page 4). We assume the size of the activations doubles at
        each layer (so image size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            img_size:
                the size of the image, i.e. the input of the discriminator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the discriminator (starting
                closest to the middle of the DCGAN and going outward, i.e. in reverse-
                chronological order for the discriminator)
        """
        n_layers = len(hidden_channels)
        assert img_size % (2**n_layers) == 0, "activation size must double at each layer"

        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels

        in_channels = [img_channels] + hidden_channels[:-1]
        out_channels = hidden_channels
        conv_layers = []

        for i, (ic, oc) in enumerate(zip(in_channels,out_channels)):
            if i == 0:
                conv_layers.extend([Conv2d(ic,oc,4,2,1), LeakyReLU(0.2) ])
            else:
                conv_layers.extend([Conv2d(ic,oc,4,2,1), BatchNorm2d(oc), LeakyReLU(0.2)])

        self.hidden_layers = Sequential(*conv_layers)

        final_height = img_size // (2**n_layers)
        final_size = hidden_channels[-1] * (final_height**2)
        self.classifier = Sequential(
            Rearrange("b c h w-> b (c h w)"),
            Linear(final_size, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x.squeeze(1)


class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.netD = Discriminator(img_size, img_channels, hidden_channels)
        self.netG = Generator(latent_dim_size, img_size, img_channels, hidden_channels)

        initialize_weights(self.netD)
        initialize_weights(self.netG)

@dataclass
class DCGANArgs:
    """
    Class for the arguments to the DCGAN (training and architecture).
    Note, we use field(defaultfactory(...)) when our default value is a mutable object.
    """

    # architecture
    latent_dim_size: int = 100
    hidden_channels: list[int] = field(default_factory=lambda: [128, 256, 512])

    # data & training
    dataset: Literal["MNIST", "CELEB"] = "CELEB"
    batch_size: int = 64
    epochs: int = 3
    lr: float = 0.0002
    betas: tuple[float, float] = (0.5, 0.999)
    clip_grad_norm: float | None = 1.0

    # logging
    use_wandb: bool = False
    wandb_project: str | None = "day5-gan"
    wandb_name: str | None = None
    log_every_n_steps: int = 250


class DCGANTrainer:
    def __init__(self, args: DCGANArgs):
        self.args = args
        self.trainset = get_dataset(self.args.dataset)
        self.trainloader = DataLoader(
            self.trainset, batch_size=args.batch_size, shuffle=True, num_workers=8
        )

        batch, img_channels, img_height, img_width = next(iter(self.trainloader))[0].shape
        assert img_height == img_width

        self.model = (
            DCGAN(args.latent_dim_size, img_height, img_channels, args.hidden_channels)
            .to(device)
            .train()
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.optG = t.optim.Adam(self.model.netG.parameters(), lr=args.lr, betas=args.betas)
        self.optD = t.optim.Adam(self.model.netD.parameters(), lr=args.lr, betas=args.betas)

    def training_step_discriminator(
        self,
        img_real: Float[Tensor, "batch channels height width"],
        img_fake: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, ""]:
        """
        Generates a real and fake image, and performs a gradient step on the discriminator to
        maximize log(D(x)) + log(1-D(G(z))). Logs to wandb if enabled.
        """
        self.optD.zero_grad()

        
        # Forward pass
        output_real = self.model.netD(img_real)
        output_fake = self.model.netD(img_fake)

        real_t = t.full_like(output_real, 0.9)  # 0.9 instead of 1.0
        fake_t = t.zeros_like(output_fake)

        lossD = self.loss(output_real, real_t) + self.loss(output_fake, fake_t)
        
        lossD.backward()
        if self.args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.netD.parameters(), self.args.clip_grad_norm)

        self.optD.step()
        
        if self.args.use_wandb:
            wandb.log({"lossD": lossD}, step=self.step)#
        return lossD

    def training_step_generator(
        self, img_fake: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, ""]:
        """
        Performs a gradient step on the generator to maximize log(D(G(z))). Logs to wandb if enabled.
        """
        self.optG.zero_grad()
        # Forward pass on discriminator
        output = self.model.netD(img_fake)
        real_t = t.full_like(output,1)
        
        lossG = self.loss(output, real_t)  
        lossG.backward()

        if self.args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.netG.parameters(), self.args.clip_grad_norm)

        self.optG.step()
        if self.args.use_wandb:
            wandb.log({"lossG": lossG}, step=self.step)
        return lossG

    @t.inference_mode()
    def log_samples(self) -> None:
        """
        Performs evaluation by generating 8 instances of random noise and passing them through the
        generator, then optionally logging the results to Weights & Biases.
        """
        assert self.step > 0, (
            "First call should come after a training step. Remember to increment `self.step`."
        )
        self.model.netG.eval()

        # Generate random noise
        t.manual_seed(42)
        noise = t.randn(10, self.model.latent_dim_size).to(device)
        # Get generator output
        output = self.model.netG(noise)
        # Clip values to make the visualization clearer
        output = output.clamp(output.quantile(0.01), output.quantile(0.99))
        # Log to weights and biases
        if self.args.use_wandb:
            output = einops.rearrange(output, "b c h w -> b h w c").cpu().numpy()
            wandb.log({"images": [wandb.Image(arr) for arr in output]}, step=self.step)
        else:
            display_data(output, nrows=1, title="Generator-produced images")

        self.model.netG.train()

    def train(self) -> DCGAN:
        """Performs a full training run."""
        self.step = 0
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)

        for epoch in range(self.args.epochs):
            progress_bar = tqdm(self.trainloader, total=len(self.trainloader), ascii=True)

            for img_real, label in progress_bar:
                z = t.randn(self.args.batch_size, self.model.latent_dim_size).to(device)
                img_fake = self.model.netG(z)
                img_real = img_real.to(device)
                lossD = self.training_step_discriminator(img_real, img_fake.detach())
                lossG = self.training_step_generator(img_fake)
                self.step += 1

                progress_bar.set_description(f"{epoch=}, {lossD=:.4f}, {lossG=:.4f}, batches={self.step}")

                if self.step % self.args.log_every_n_steps == 0:
                    self.log_samples()
            #make dir models
            os.makedirs("models", exist_ok=True)
            #save model weights at the end of each epoch
            t.save(self.model.state_dict(), f"models/epoch_{epoch}_dcgan.pth")

        if self.args.use_wandb:
            wandb.finish()

        return self.model