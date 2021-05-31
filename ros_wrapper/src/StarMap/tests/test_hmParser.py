from pathlib import Path
from operator import truediv
from functools import reduce

import torch
import torch.nn
import numpy as np
import cv2

from starmap.utils.hmParser import nms, parseHeatmap

def absfilename(reldirs,
                reffile=__file__):
    return reduce(truediv, reldirs, Path(reffile).parent)

LENNA_IMG = absfilename(['data' , 'lena512.pgm'])

def safe_cv2_imread(f, *args, **kw):
    ret = cv2.imread(f, *args, **kw)
    if ret is None:
        raise IOError("Unable to open file : {}".format(f))
    return ret

def test_nms(infname= LENNA_IMG,
             expectedfname=absfilename(['data', 'test-lenna-nms-out.pgm'])):
    pool = (
        nms(safe_cv2_imread(str(infname), cv2.IMREAD_UNCHANGED) / 255.) * 255.
    ).astype(np.uint8)
    assert (pool == cv2.imread(str(expectedfname), cv2.IMREAD_UNCHANGED)).all()


class Container(torch.nn.Module):
    def __init__(self, ptsidx):
        super().__init__()
        self.ptsidx = ptsidx

    def forward(self, x):
        return self.ptsidx * x


def test_parseHeatmap(infname=LENNA_IMG,
                      expectedfname=absfilename(['data', 'test-lenna-parseHeatmap-out.cv2.yaml'])):
    img = safe_cv2_imread(str(infname), cv2.IMREAD_UNCHANGED) / 255.
    img = img[None, ...]
    hm = np.concatenate((img, img, img))
    pts = parseHeatmap(hm)
    ptsidx = np.vstack(pts)
    # Serialize using h5py
    # with h5py.File(expectedfname, 'w') as f:
    #     f["ptsidx"] = ptsidx
    # assert (ptsidx == h5py.File(expectedfname, 'r')["ptsidx"]).all()

    # Serialize using torch
    # Save arbitrary values supported by TorchScript
    # https://pytorch.org/docs/master/jit.html#supported-type
    # container = torch.jit.script(
    #     Container(ptsidx=torch.from_numpy(ptsidx))
    # )
    # container.save(str(expectedfname))
    # assert (ptsidx == torch.load(expectedfname).detach().numpy()).all()

    # Serialize using opencv
    # fs = cv2.FileStorage(str(expectedfname), cv2.FileStorage_WRITE_BASE64)
    # fs.write("ptsidx", ptsidx)
    # fs.release()
    fs = cv2.FileStorage(str(expectedfname), cv2.FileStorage_READ)
    expected_pts = fs.getFirstTopLevelNode().mat();
    assert (ptsidx == expected_pts).all()

if __name__ == '__main__':
    test_parseHeatmap()

