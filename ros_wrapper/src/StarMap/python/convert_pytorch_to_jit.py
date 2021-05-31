from pathlib import Path

import torch

from models.hg import HourglassNet

__directory__ = Path(__file__).parent or "."

def convert_pytorch_to_jit(
        frompath=__directory__ / '..' / 'models' / 'model_cpu.pth',
        topath_gen=lambda frompath: str(frompath).replace(".pth", "-jit.pth")):
    """
    https://pytorch.org/tutorials/advanced/cpp_export.html
    """
    model = torch.load(str(frompath))
    model_new = HourglassNet(model.nStack, model.nModules, model.nFeats,
                             model.numOutput)
    model_new.load_state_dict(model.state_dict())
    example = torch.rand(1, 3, 256, 256)
    traced_script_module = torch.jit.trace(model_new, example)
    torch.jit.save(traced_script_module, topath_gen(frompath))

if __name__ == '__main__':
    convert_pytorch_to_jit()
