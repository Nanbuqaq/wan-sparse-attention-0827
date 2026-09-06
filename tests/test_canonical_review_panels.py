from PIL import Image
from scripts.build_canonical_review_panels import retile_quarter


def test_retile_keeps_every_thumbnail_pixel_without_aspect_distortion():
    board = Image.new('RGB', (1664, 240))
    for slot in range(16):
        tile = Image.new('RGB', (208, 120), (slot*13, 100, 50))
        tile.putpixel((17, 91), (255, 255, 255))
        board.paste(tile, ((slot%8)*208, (slot//8)*120))
    result = retile_quarter(board)
    assert result.size == (832, 480)
    for slot in range(16):
        x, y = (slot%4)*208, (slot//4)*120
        assert result.getpixel((x, y)) == (slot*13, 100, 50)
        assert result.getpixel((x+17, y+91)) == (255, 255, 255)
