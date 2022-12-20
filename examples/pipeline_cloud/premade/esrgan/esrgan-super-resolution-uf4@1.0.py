from typing import Any, TypedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor, nn

from pipeline import (
    Pipeline,
    PipelineCloud,
    PipelineFile,
    Variable,
    pipeline_function,
    pipeline_model,
)

"""
Run the following command to download weights

wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
mv RealESRGAN_x4plus.pth esrgan.pt


"""
checkpoint_path = "851b069b5f1967014bb11a69081cc0c5"


def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type
    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.
    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV
    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=False, half=False)
    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = (
        tensor.squeeze(0)
        .permute(1, 2, 0)
        .mul(255)
        .clamp(0, 255)
        .cpu()
        .numpy()
        .astype("uint8")
    )

    return image


def image_to_tensor(image: ndarray, range_norm: bool, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch
    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type
    Returns:
        tensor (Tensor): Data types supported by PyTorch
    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=True, half=False)
    """
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def preprocess_one_image(image_path: str, device: torch.device) -> Tensor:
    image = cv2.imread(image_path).astype(np.float32) / 255.0

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image data to pytorch format data
    tensor = image_to_tensor(image, False, False).unsqueeze_(0)

    print(tensor.size())  # 1, 3, 512, 512

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(
        device=device, memory_format=torch.channels_last, non_blocking=True
    )

    return tensor


class _ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1)
        )
        self.conv2 = nn.Conv2d(
            channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1)
        )
        self.conv3 = nn.Conv2d(
            channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1)
        )
        self.conv4 = nn.Conv2d(
            channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1)
        )
        self.conv5 = nn.Conv2d(
            channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1)
        )

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out


class _ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 64,
        growth_channels: int = 32,
        num_blocks: int = 23,
        upscale_factor: int = 4,
    ) -> None:
        print("A1/1 loading RRDBNet")
        super(RRDBNet, self).__init__()
        self.upscale_factor = upscale_factor

        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        if upscale_factor == 2:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
        if upscale_factor == 4:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
        if upscale_factor == 8:
            self.upsampling1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
            self.upsampling3 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Output layer.
        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize all layer
        self._initialize_weights()

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        if self.upscale_factor == 2:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
        if self.upscale_factor == 4:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
            out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))
        if self.upscale_factor == 8:
            out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
            out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))
            out = self.upsampling3(F.interpolate(out, scale_factor=2, mode="nearest"))

        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class PromptShape(TypedDict):
    image_in: str


@pipeline_model
class ESRGANModel:
    @pipeline_function
    def predict(self, prompts: list[PromptShape]) -> list[list[str]]:
        print("🦁 run start")
        import base64
        from io import BytesIO

        from PIL import Image

        for index, prompt in enumerate(prompts):
            if not len(prompt["image_in"]) > 0:
                raise ValueError(f"Input image at prompt index {index} is invalid.")

        all_outputs = []

        with torch.no_grad():
            for index, prompt in enumerate(prompts):
                print(f"🌀 prompt {index}/{len(prompts) - 1}")

                input_image = base64.b64decode(prompt["image_in"])
                input_buffer = BytesIO(input_image)
                input_image = Image.open(input_buffer)
                input_image = input_image.convert("RGB")
                input_image = torch.from_numpy(
                    (np.array(input_image).astype(np.float32) / 255.0)[None].transpose(
                        0, 3, 1, 2
                    )
                ).to(0)

                prompt_images = []

                sr_tensor = self.model(input_image)

                sr_image = (
                    sr_tensor.squeeze(0)
                    .permute(1, 2, 0)
                    .mul(255)
                    .clamp(0, 255)
                    .cpu()
                    .numpy()
                    .astype("uint8")
                )
                sr_image = Image.fromarray(sr_image)

                buffered = BytesIO()
                sr_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                prompt_images.append(img_str)

                metadata = {
                    "upscale_factor": 4,
                }

                prompt_dict = {
                    "samples": prompt_images,
                    "metadata": metadata,
                }
                all_outputs.append(prompt_dict)

                print(f"✅ prompt {index}/{len(prompts) - 1}")

        print("🐸 run complete")
        return all_outputs

    @pipeline_function(run_once=True, on_startup=True)
    def load(self, checkpoint_file: PipelineFile) -> bool:

        # it would be lovely to pass `device` to this load function, but for now...
        device = torch.device("cuda:0")

        self.model = (
            RRDBNet(
                upscale_factor=4,  # parameterise the 4, can be 1, 2, 4, 8
                in_channels=3,
                out_channels=3,
                channels=64,
                growth_channels=32,
                num_blocks=23,
            )
            .to(device)
            .eval()
        )

        print("B1/1 loading weights")
        state_dict = torch.load(checkpoint_file.path, map_location=device)

        print("C1/1 loading weights into model")
        self.model.load_state_dict(state_dict)

        print("🥑 model and weights load complete")

        return True


with Pipeline("ESRGAN Super Resolution", min_gpu_vram_mb=10000) as builder:
    prompts = Variable(list, is_input=True)
    checkpoint_file = PipelineFile(path=checkpoint_path)

    builder.add_variables(prompts, checkpoint_file)

    esrgan_model = ESRGANModel()
    esrgan_model.load(checkpoint_file)

    output = esrgan_model.predict(prompts)
    builder.output(output)

new_pipeline = Pipeline.get_pipeline("ESRGAN Super Resolution")

upload = False
if upload:
    api = PipelineCloud()
    uploaded_pipeline = api.upload_pipeline(new_pipeline)
    print(f"Uploaded pipeline id: {uploaded_pipeline.id}")
else:
    import base64
    import urllib.request
    from io import BytesIO

    from PIL import Image

    url = "https://paulcjh.com/robot.jpeg"
    urllib.request.urlretrieve(url, "gfg.jpg")

    img = Image.open("gfg.jpg")

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    result = new_pipeline.run([dict(image_in=img_str)])[0]["samples"][0]
    print(result)
