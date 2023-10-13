import os
from pathlib import Path

from pipeline import File, Pipeline, Variable, entity, pipe
from pipeline.cloud import compute_requirements, pipelines
from pipeline.cloud.environments import create_environment


@entity
class RealESRGAN:
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(
        self,
        scale: int = 4,
        pre_pad: int = 0,
        # use full precision by default
        half=False,
    ):
        self.scale = scale
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half
        self.model = None

    @pipe(on_startup=True, run_once=True)
    def load(self, model_file: File) -> None:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loadnet = torch.load(model_file.path, map_location=torch.device("cpu"))

        # prefer to use params_ema
        if "params_ema" in loadnet:
            keyname = "params_ema"
        else:
            keyname = "params"

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def pre_process(self, img):
        import numpy as np
        import torch
        from torch.nn import functional as F

        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible"""
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if h % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - h % self.mod_scale
            if w % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - w % self.mod_scale
            self.img = F.pad(
                self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect"
            )

    def process(self):
        # model inference
        self.output = self.model(self.img)

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.mod_pad_h * self.scale,
                0 : w - self.mod_pad_w * self.scale,
            ]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.pre_pad * self.scale,
                0 : w - self.pre_pad * self.scale,
            ]
        return self.output

    def _enhance(self, img, outscale=None, alpha_upsampler="realesrgan"):
        import cv2
        import numpy as np
        import torch

        with torch.no_grad():
            h_input, w_input = img.shape[0:2]
            # img: numpy
            img = img.astype(np.float32)
            if np.max(img) > 256:  # 16-bit image
                max_range = 65535
                print("\tInput is a 16-bit image")
            else:
                max_range = 255
            img = img / max_range
            if len(img.shape) == 2:  # gray image
                img_mode = "L"
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA image with alpha channel
                img_mode = "RGBA"
                alpha = img[:, :, 3]
                img = img[:, :, 0:3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if alpha_upsampler == "realesrgan":
                    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
            else:
                img_mode = "RGB"
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ------------------- process image (without the alpha channel) ------------------- #
            self.pre_process(img)
            self.process()
            output_img = self.post_process()
            output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
            if img_mode == "L":
                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

            # ------------------- process the alpha channel if necessary ------------------- #
            if img_mode == "RGBA":
                if alpha_upsampler == "realesrgan":
                    self.pre_process(alpha)
                    self.process()
                    output_alpha = self.post_process()
                    output_alpha = (
                        output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    )
                    output_alpha = np.transpose(
                        output_alpha[[2, 1, 0], :, :], (1, 2, 0)
                    )
                    output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
                else:  # use the cv2 resize for alpha channel
                    h, w = alpha.shape[0:2]
                    output_alpha = cv2.resize(
                        alpha,
                        (w * self.scale, h * self.scale),
                        interpolation=cv2.INTER_LINEAR,
                    )

                # merge the alpha channel
                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
                output_img[:, :, 3] = output_alpha

            # ------------------------------ return ------------------------------ #
            if max_range == 65535:  # 16-bit image
                output = (output_img * 65535.0).round().astype(np.uint16)
            else:
                output = (output_img * 255.0).round().astype(np.uint8)

            if outscale is not None and outscale != float(self.scale):
                output = cv2.resize(
                    output,
                    (
                        int(w_input * outscale),
                        int(h_input * outscale),
                    ),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            return output, img_mode

    @pipe
    def enhance(self, image: File, outscale: int) -> File:
        import cv2

        # if args.face_enhance:  # Use GFPGAN for face enhancement
        #     from gfpgan import GFPGANer
        #     face_enhancer = GFPGANer(
        #         model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        #         upscale=args.outscale,
        #         arch="clean",
        #         channel_multiplier=2,
        #         bg_upsampler=upsampler,
        #     )

        print(f"Enhancing image {image.path}")
        imgname, extension = os.path.splitext(os.path.basename(image.path))

        #  img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = "RGBA"
        else:
            img_mode = None

        # if args.face_enhance:
        #     _, _, output = face_enhancer.enhance(
        #         img, has_aligned=False, only_center_face=False, paste_back=True
        #     )
        # else:
        #     output, _ = upsampler.enhance(img, outscale=args.outscale)
        output, _ = self._enhance(img, outscale=outscale)
        extension = extension[1:]
        if img_mode == "RGBA":  # RGBA images should be saved in png format
            extension = "png"

        save_path = Path(f"/tmp/real_esrgan/image_{imgname}_out.{extension}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), output)
        return File(path=save_path, allow_out_of_context_creation=True)


with Pipeline() as builder:
    image = Variable(
        File,
        title="Image File",
        description="Upload a .png, .jpg or other image file to be enhanced.",
    )
    outscale = Variable(
        int,
        default=4,
        ge=1,
        le=10,
        title="Output Scale",
        description="Factor to scale image by",
    )

    model = RealESRGAN()
    model_file = File(path="RealESRGAN_x4plus.pth")
    model.load(model_file)

    output = model.enhance(image, outscale)
    builder.output(output)

my_pl = builder.get_pipeline()

env_name = "ross/real-esrgan"

env_id = create_environment(
    name=env_name,
    python_requirements=[
        "basicsr==1.4.2",
        "facexlib==0.3.0",
        "gfpgan==1.3.8",
        "numpy==1.24.4",
        "opencv-python==4.8.0.76",
        "Pillow==10.0.1",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "tqdm==4.66.1",
    ],
)


pipelines.upload_pipeline(
    my_pl,
    "ross/real-esrgan",
    environment_id_or_name=env_name,
    required_gpu_vram_mb=16_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_all,
    ],
)
