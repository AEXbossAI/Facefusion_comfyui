import os
import torch
import numpy as np


class FaceEnhanceNode:
    """Enhance/restore face region using GFPGAN after face swap."""

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

        # standard ComfyUI location
        model_path = "/opt/ComfyUI/models/facerestore_models/GFPGANv1.4.pth"
        if not os.path.exists(model_path):
            # fallback: relative from this file up to ComfyUI root
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
            model_path = os.path.join(root, "models", "facerestore_models", "GFPGANv1.4.pth")

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
