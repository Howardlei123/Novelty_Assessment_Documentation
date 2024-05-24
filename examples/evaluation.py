from typing import Protocol

class Evaluation(Protocol):
    def switch(self,run) -> None:
        ...

class Histogram: # to find the best window size using an initial threshold value
    def switch(self.run) -> None: