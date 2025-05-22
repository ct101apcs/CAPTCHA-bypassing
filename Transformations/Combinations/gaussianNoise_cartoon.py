from Transformations.GaussianNoise.gaussianNoise import gaussianNoise
from Transformations.Cartoon.cartoon import cartoon
from .base_combination import BaseCombination

def gaussianNoise_cartoon(img, config_params=None):
    combination = BaseCombination([
        (gaussianNoise, {}),
        (cartoon, {})
    ], config_params)
    return combination(img) 