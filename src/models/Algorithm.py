from abc import ABC, abstractmethod

class Algorithm(ABC):
    @abstractmethod
    def __init__(self, func):
        self.func, self.args, self.kwargs = func
        self.alg = []
        self.build_algorithm()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        return self.func(*self.args, **kwargs)

    @abstractmethod
    def __str__(self):
        return str(self.alg)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Algorithm:
            if any("__call__" in B.__dict__ for B in C.__mro__):
                if any("__str__" in B.__dict__ for B in C.__mro__):
                    return True
        return NotImplemented

    @abstractmethod
    def build_algorithm(self):
        self.alg = []



