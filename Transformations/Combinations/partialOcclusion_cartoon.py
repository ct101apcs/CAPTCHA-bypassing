from Transformations.PartialOcclusion.partialOcclusion import partialOcclusion
from Transformations.Cartoon.cartoon import cartoon
from .base_combination import BaseCombination

def partialOcclusion_cartoon(img, config_params=None):
    combination = BaseCombination([
        (partialOcclusion, {}),
        (cartoon, {})
    ], config_params)
    return combination(img) 