# digitalDigitRecognition

This is a basic image classifier built on top of an MNIST implementation in Python, using Tensorflow. 
Image inputs are hand-signed digits gestures. On a toy data set consisting of gestures with varying angle
and image placements on a simple background, the network acheives > 96% accuracy in less than 15000 steps of training.

Examples of the input images and their classifications chosen by the network are below:

<img src="https://github.com/theScinder/digitalDigitRecognition/blob/master/toyModelDemo0123_1.png">


<img src="https://github.com/theScinder/digitalDigitRecognition/blob/master/toyModelDemo0123_2.png">

All of the above are correctly recognized, but for the ~4% initial errors the main source of confusion seems to be a gesture for 
"two" using the thumb and forefinger in some of the images. 
This looks similar to the gesture for "three," and I'll update the network to better account for similar gestures in the next iteration. 

A typical call with the following flags is a good starting point: 

```
python handToy2x4.py --maxSteps=15000 --dropout=0.5 -learningRate=1e-3 --batchSize=256
```
