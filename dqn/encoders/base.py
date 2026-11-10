from abc import ABC, abstractmethod
import numpy as np


class StateEncoder(ABC):

    @abstractmethod
    def encode(self, env) -> np.ndarray:
        """Convert the current environment state into a numerical array."""
        pass

    @property
    @abstractmethod
    def input_shape(self) -> tuple:
        """Return the shape expected by the neural network."""
        pass