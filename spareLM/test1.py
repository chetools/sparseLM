

N=20
Nx=5

import numpy as np

rng=np.random.RandomState(123)


colsets=rng.choice(np.arange(N),size=(N,Nx))
combos=set(colsets)



