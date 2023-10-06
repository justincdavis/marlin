# marlin

Implementation of the methods described in the paper:
Frugal Following: Power Thrifty Object Detection and Tracking for Mobile Augmented Reality
https://dl.acm.org/doi/pdf/10.1145/3356250.3360044

To install simply clone and run
`pip3 install .`

Once installed to use:
```
from marlin import Marlin

dnn = ... # (a callable function which returns a list of detections)

marlin = Marlin(dnn)

for frame in videofile:
  dections = marlin(frame)
```
