
import numpy as np
from numpy.typing import NDArray

name = "Kante Ngolo"
print(f"Hello {name}")

class Aircraft:
    def __init__(self, velocity: NDArray[np.float64]) -> None:
        self.velocity = velocity

print("This is perfect!")