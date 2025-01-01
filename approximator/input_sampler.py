from abc import ABC, abstractmethod
from typing import List, Any


class InputSampler(ABC):
    """
    The InputSampler yields a batch of samples. The user can provide their own implementations of this
    to sufficiently represent the input space of their function.
    """

    @abstractmethod
    def sample(self, batch_size: int) -> List[List[Any]]:
        """
        Sample a batch of inputs
        :param batch_size: the number of samples to return
        :return: a batch of samples (list of lists where first dim is the number of samples, second is number of inputs)
        """

        raise NotImplementedError()
