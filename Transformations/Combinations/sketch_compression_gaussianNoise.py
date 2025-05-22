from Transformations.Sketch.sketch import sketch
from Transformations.Compression.compression import compression
from Transformations.GaussianNoise.gaussianNoise import gaussianNoise
from .base_combination import BaseCombination

def sketch_compression_gaussianNoise(img, config_params=None):
    combination = BaseCombination([
        (sketch, {}),
        (compression, {}),
        (gaussianNoise, {})
    ], config_params)
    return combination(img) 