import torch
import torch.jit

import utils

# NOTE torchscript: https://pytorch.org/docs/master/jit.html
# NOTE torchscript with C++: https://pytorch.org/tutorials/advanced/cpp_export.html
# NOTE towardsdatascience: https://towardsdatascience.com/pytorch-jit-and-torchscript-c2a77bac0fff

def build_inference_fn() -> torch.jit.ScriptFunction:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    traced_fn = torch.jit.trace(lambda x: utils.infer_frame(utils.load_model(device), x, device), torch.rand((1000,1000,3)))
    return traced_fn

def serialize_inference_fn() -> None:
    fn = build_inference_fn()
    fn.save("traced_model_fn.pth")