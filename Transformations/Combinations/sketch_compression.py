from Transformations.Sketch.sketch import sketch
from Transformations.Compression.compression import compression
from .base_combination import BaseCombination

def sketch_compression(img, config_params=None):
    combination = BaseCombination([
        (sketch, {}),
        (compression, {})
    ], config_params)
    return combination(img) 