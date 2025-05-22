from Transformations.Sketch.sketch import sketch
from Transformations.Swirl.swirl import swirl
from Transformations.GaussianNoise.gaussianNoise import gaussianNoise
from .base_combination import BaseCombination

def sketch_swirl_gaussianNoise(img, config_params=None):
    combination = BaseCombination([
        (sketch, {}),
        (swirl, {}),
        (gaussianNoise, {})
    ], config_params)
    return combination(img) 