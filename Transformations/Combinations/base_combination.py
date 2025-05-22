from Transformations.utils import apply_transformation
import json

class BaseCombination:
    def __init__(self, transformations, config_params=None):
        """
        Initialize with a list of (transform_func, params) tuples
        If config_params is provided, use those instead of the default params
        """
        self.transformations = transformations
        self.config_params = config_params

    def __call__(self, img):
        """
        Apply all transformations in sequence
        """
        current_img = img
        for transform_func, _ in self.transformations:
            # Get the function name from the module
            func_name = transform_func.__name__
            # Use parameters from config if available, otherwise use default
            params = self.config_params.get(func_name, {}) if self.config_params else {}
            current_img = apply_transformation(current_img, transform_func, **params)
        return current_img 