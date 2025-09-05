import numpy as np
import torch
from typing import List, Optional, Sequence, Union

class SapiensPoseInference:
    def __init__(
        self, 
        device: torch.device,
        dtype: torch.dtype,
        pose_checkpoint: str
    ):
        self.device = device
        self.dtype = dtype
        self.model = torch.jit.load(pose_checkpoint).eval().to(device).to(dtype)

    def warmup_model(model, batch_size):
        # Warm up the model with a dummy input.
        imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=torch.bfloat16).cuda()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            for i in range(3):
                model(imgs)
        torch.cuda.current_stream().wait_stream(s)
        imgs = imgs.detach().cpu().float().numpy()
        del imgs, s


    @torch.inference_mode()
    def pose(self, inputs: List[Union[np.ndarray, str]]):
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self.dtype):
            heatmaps = model(imgs.cuda())
            imgs.cpu()
        return heatmaps.cpu()