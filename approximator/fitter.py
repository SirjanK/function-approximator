from approximator.input_sampler import InputSampler
from abc import ABC, abstractmethod
from typing import Callable, List, Any, Tuple


class Fitter(ABC):
    """
    The Fitter class allows a user to custom implement the logic to fit a function to the approximate a certain input function given
    an input sampler and the function itself.
    """

    def __init__(self, fn: Callable, input_sampler: InputSampler):
        self._fn = fn
        self._input_sampler = input_sampler
        
    @abstractmethod
    def fit(self) -> Callable:
        """
        Fit the function

        :return the approximated function
        """

        raise NotImplementedError()
    
    def _generate_batch(self, batch_size: int) -> Tuple[List[List[Any]], List[Any]]:
        """
        Generate a batch of samples
        :param batch_size: the number of samples to generate
        :return: a batch of samples with corresponding outputs
        """

        inputs = self._input_sampler.sample(batch_size)
        outputs = [self._fn(*inp) for inp in inputs]

        return inputs, outputs
