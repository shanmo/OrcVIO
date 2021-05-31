from pathlib import Path
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s %(funcName)-15s %(message)s')
LOG = logging.getLogger(__name__)

import cv2
import numpy as np

from starmap.utils.img import Crop, NewCrop

DEBUG = False

def random_img(img_shape):
    img = np.random.randint(255, size=(img_shape[0], img_shape[1], 3), dtype=np.uint8)
    return img

def imread_img(img_shape,
               fname=Path(__file__).parent / 'data' / 'lena512.pgm'):
    return cv2.resize(cv2.imread(str(fname)), img_shape)


def _test_crop(img_shape, desired_side=256, rot=0, img_gen=imread_img,
               new_crop_kw={}, skip_match=False):
    """
    @param img_shape: (height, width)
    """
    img = img_gen(img_shape)
    center = np.array([img.shape[1] // 2, img.shape[0] // 2])
    max_side = max(img.shape[:2])
    new_img = Crop(img, center, max_side, 0, desired_side)
    proposed_img = NewCrop(img, center, max_side, 0, desired_side, **new_crop_kw)
    if DEBUG and not (new_img != proposed_img).sum() / np.prod(new_img.shape) < 0.02:
        cv2.imshow("Crop", new_img)
        cv2.imshow("NewCrop", proposed_img)
        cv2.imshow("diff", new_img - proposed_img)
        cv2.waitKey(-1)
        LOG.warning("Images did not match with %d pixels" % np.sum(new_img != proposed_img))
        #assert False
    elif not skip_match:
        assert (new_img != proposed_img).sum() / np.prod(new_img.shape) < 0.02


def test_crop_lena():
    _test_crop((300, 400), img_gen=imread_img)


def test_crop_fat_big_big_no_rot():
    _test_crop((300, 400))


def test_crop_fat_small_big_no_rot():
    _test_crop((100, 400))


def test_crop_fat_small_small_no_rot():
    _test_crop((100, 200))


def test_crop_tall_small_small_no_rot(new_crop_kw=dict()):
    _test_crop((200, 100), new_crop_kw=new_crop_kw)


def test_crop_tall_big_small_no_rot():
    _test_crop((400, 100))


def test_crop_tall_big_big_no_rot():
    _test_crop((400, 300))


def test_crop_tall_big_small_no_rot():
    _test_crop((800, 100))


def test_crop_tall_big_big_no_rot():
    _test_crop((800, 750))

def test_crop_odd_size():
    _test_crop((213, 304), skip_match=True)


if __name__  == '__main__':
    ## Succeeds
    test_crop_lena()
    test_crop_fat_big_big_no_rot()
    test_crop_fat_small_big_no_rot()
    test_crop_fat_small_small_no_rot()
    test_crop_tall_small_small_no_rot()
    test_crop_tall_big_small_no_rot()
    test_crop_tall_big_big_no_rot()
    test_crop_tall_big_small_no_rot()
    test_crop_tall_big_big_no_rot()
    test_crop_odd_size()
