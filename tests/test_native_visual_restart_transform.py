import pytest


def crop_box(width, height):
    cw, ch = int(width*.75), int(height*.75)
    left, top = (width-cw)//2, (height-ch)//2
    return (left, top, left+cw, top+ch)


def test_crop_is_fixed_fraction_and_centered():
    assert crop_box(1280, 704) == (160, 88, 1120, 616)


def test_crop_does_not_change_geometry():
    b = crop_box(1280, 704)
    assert b[2]-b[0] == 960 and b[3]-b[1] == 528
