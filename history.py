import random
from collections import deque
from dataclasses import dataclass


@dataclass
class Sample:
    """ a single sample in the replay buffer, which can be serialized
    and sent back from a data gathering agent to the learning process """
    state: list = None
    action: int = None
    reward: float = None
    next_state: list = None
    done: bool = None


class History:
    """ a list of samples that we can randomly sample for learning """
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add_sample(self, sample: Sample):
        """ adds a sample, while discarding old entries

        :param sample: a single Sample to be added to the buffer
        """
        self.buffer.append(sample)

    def random_sample(self, batch_size: int):
        """
        :param batch_size: how many elements to return from the buffer
        :returns: a list of Sample
        """
        samples = random.sample(self.buffer, batch_size)
        return samples

    def __len__(self):
        return len(self.buffer)
