import pytest
from src.preprocess import VideoReader


def test_videoopen():
    # basic smoke: open a non-existent file must raise
    import os
    with pytest.raises(RuntimeError):
        VideoReader('nonexistent.mp4')
