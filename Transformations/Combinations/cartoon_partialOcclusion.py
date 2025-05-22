from Transformations.Cartoon.cartoon import cartoon
from Transformations.PartialOcclusion.partialOcclusion import partialOcclusion
from .base_combination import BaseCombination

def cartoon_partialOcclusion(img, config_params=None):
    combination = BaseCombination([
        (cartoon, {}),
        (partialOcclusion, {})
    ], config_params)
    return combination(img) 