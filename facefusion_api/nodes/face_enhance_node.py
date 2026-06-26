import os
import torch
import numpy as np


def _patch_basicsr():
    """Fix basicsr compatibility with torchvision>=0.16 (functional_tensor removed)."""
    try:
        import basicsr
        deg_path = os.path.join(os.path.dirname(basicsr.__file__), "data", "degradations.py")
        if os.path.exists(deg_path):
            with open(deg_path) as f:
                src = f.read()
            old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
            new = "from torchvision.transforms.functional import rgb_to_grayscale"
            if old in src:
                with open(deg_path, "w") as f:
                    f.write(src.replace(old, new))
                print("[FaceEnhance] patched basicsr/data/degradations.py")
    except Exception as e:
        print(f"[FaceEnhance] basicsr patch skipped: {e}")


_patch_basicsr()


class FaceEnhanceNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":  ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "FaceFusion"

    def enhance(self, image, weight):
        try:
            import cv2
            from gfpgan import GFPGANer
        except ImportError as e:
            print(f"[FaceEnhance] gfpgan not installed: {e}")
            return (image,)

        # model lives next to other FF models in custom_nodes/Facefusion_comfyui/models/
        model_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "GFPGANv1.4.pth"
        ))
        if not os.path.exists(model_path):
            model_path = "/opt/ComfyUI/models/facerestore_models/GFPGANv1.4.pth"

        if not os.path.exists(model_path):
            print(f"[FaceEnhance] GFPGANv1.4.pth not found at {model_path}")
            return (image,)

        results = []
        for i in range(image.shape[0]):
            img_np = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            try:
                restorer = GFPGANer(
                    model_path=model_path,
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                )
                _, _, restored = restorer.enhance(
                    img_bgr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=weight,
                )
                restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
                results.append(torch.from_numpy(restored_rgb.astype(np.float32) / 255.0))
            except Exception as e:
                print(f"[FaceEnhance] frame {i} error: {e}")
                results.append(image[i].cpu())

        return (torch.stack(results),)
