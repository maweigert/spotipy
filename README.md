![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spotipy-detector.svg)
![PyPI - Version](https://img.shields.io/pypi/v/spotipy-detector)
![PyPI - License](https://img.shields.io/pypi/l/spotipy-detector)
![PyPI - Status](https://img.shields.io/pypi/status/spotipy-detector)
![PyPI - Downloads](https://img.shields.io/pypi/dm/spotipy-detector)

![Logo](artwork/spotipy_transp_small.png)

---

# Spotipy - Accurate and efficient spot detection with CNNs


## Installation 


Install the [correct tensorflow for your CUDA version](https://www.tensorflow.org/install/source#gpu). 


Then Spotipy can be installed directly via `pip`:

```
pip install spotipy-detector
```


## Usage 


A `SpotNet` spot detection model can be instantiated from a custom `Config` class:


```python 

from spotipy.model import Config, SpotNet

config = Config(
        n_channel_in=1,
        unet_n_depth=2,
        train_learning_rate=3e-4,
        train_patch_size=(128,128),
        train_batch_size=4
    )

model = SpotNet(config,name="mymodel", basedir="models")

```

### Training 

The training data for a  `SpotNet` model consists of input image `X` and spot coordinates `P` (in `y,x` order):

```python 

import numpy as np
from spotipy.utils import points_to_prob

# generate some dummy data 
def dummy_data(n_samples=16):
    X = np.random.uniform(0,1,(n_samples, 128, 128))
    P = np.random.randint(0,128,(n_samples, 21, 2))
    for x, p in zip(X, P):
        x[tuple(p.T.tolist())] = np.random.uniform(2,5,len(p))
    Y = np.stack(tuple(points_to_prob(p[:,::-1], (128,128)) for p in P))
    return X, Y

X,Y = dummy_data(128)
Xv,Yv = dummy_data(16)

model.train(X,Y, validation_data=[X, Y], epochs=10, steps_per_epoch=128)

model.optimize_thresholds(Xv,Yv)

```

### Inference

Applying a trained `SpotNet`:


```python

img = dummy_data(1)[0][0]

prob, points = model.predict(img)

```


## Contributors

Albert Dominguez Mantes, Antonio Herrera, Irina Khven, Anjali Schläppi, Gioele La Manno, Martin Weigert
