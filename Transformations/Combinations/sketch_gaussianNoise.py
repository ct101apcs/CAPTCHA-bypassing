from Transformations.Sketch.sketch import sketch
from Transformations.GaussianNoise.gaussianNoise import gaussianNoise
from .base_combination import BaseCombination

def sketch_gaussianNoise(img, config_params=None):
    combination = BaseCombination([
        (sketch, {}),
        (gaussianNoise, {})
    ], config_params)
    return combination(img) 