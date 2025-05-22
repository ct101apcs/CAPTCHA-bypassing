from Transformations.Sketch.sketch import sketch
from Transformations.PartialOcclusion.partialOcclusion import partialOcclusion
from Transformations.GaussianNoise.gaussianNoise import gaussianNoise
from .base_combination import BaseCombination

def sketch_partialOcclusion_gaussianNoise(img, config_params=None):
    combination = BaseCombination([
        (sketch, {}),
        (partialOcclusion, {}),
        (gaussianNoise, {})
    ], config_params)
    return combination(img) 