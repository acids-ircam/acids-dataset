from .base import Transform
from absl import logging
import torch


class OneHot(Transform): 
    allow_random = False
    takes_as_input = Transform.input_types.torch
    def __init__(self, n_classes: int | None = None, p: float | None = None, **kwargs):
        super().__init__(p=1.0, **kwargs)
        self.n_classes = n_classes

    def __repr__(self): 
        return "OneHot(n_classes=%d)"%self.n_classes

    def apply(self, x):
        if x.dtype != torch.long: 
            logging.warning("got dtype %s in OneHot ; may cause unregular behavior"%x.dtype)
            x = x.long()
        if self.n_classes is None: 
            current_max = torch.amax(x)
            if (current_max <= 0):
                raise ValueError("lazy initialisation failed : got value with maximum <= 0.")
            self.n_classes = current_max
        return torch.nn.functional.one_hot(x, self.n_classes)