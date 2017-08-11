# digitalDigitRecognition

This is a basic image classifier built on top of an MNIST implementation in Python, using Tensorflow. 
Image inputs are hand-signed digits gestures. On a toy data set consisting of gestures with varying angle
and image placements on a simple background, the network acheives > 96% accuracy in less than 15000 steps of training.

Examples of the input images and their classifications chosen by the network are below ):

<img src="https://github.com/theScinder/digitalDigitRecognition/blob/master/outdoorsDemoRGB0to12_1.png">


<img src="https://github.com/theScinder/digitalDigitRecognition/blob/master/outdoorsDemoRGB0to12_2.png">

Results for the latest dataset (digits 0-10 with separate categories for '2' with or without thumb and null category for no hand in frame, outdoor setting) are pretty good:

```
final results with test data (not seen during training, building of model): {'accuracy': 0.97328246, 'loss': 0.071658559, 'global_step': 413298}
```

Typical call for training:

```
python classifyDigits0To12RGB.py --maxSteps=500000 --dropout=0.5 --learningRate=3e-7 --batchSize=256
```
and cludgy call for evaluation:

```
python classifyDigits0To12RGB.py --maxSteps=1 --dropout=0.0 --learningRate=3e-7 --batchSize=256 --mySeed=3
```
