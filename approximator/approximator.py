from typing import Callable, Type
from approximator.input_sampler import InputSampler
from fitter import Fitter


class Approximator:
    """
    The approximator class provides a simple interface to approximate a provided function given an input sampler and fitter class
    """

    def __init__(self, fn: Callable, input_sampler: InputSampler, fitter_cls: Type[Fitter]):
        self._fitter = fitter_cls(fn, input_sampler)
    
    def approximate(self) -> Callable:
        """
        Approximate the function
        
        :return the approximated function
        """

        return self._fitter.fit()
