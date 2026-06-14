import nacl.utils
import numpy as np

def random_bytes(n : int) -> bytes:
    """Returns n random bytes."""
    return nacl.utils.random(n)

def fresh_rng() -> np.random.Generator:
    """Returns a fresh RNG."""
    seed = int.from_bytes(random_bytes(32), 'little')
    return np.random.default_rng(seed)

