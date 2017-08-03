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

With the data set expanded to digits 0-10, with separate categories for a '2' gesture with or without using the thumb gives somewhat worse results on the test data: 

final results with test data (not seen during training, building of model): {'accuracy': 0.69444442, 'loss': 5.7782984, 'global_step': 27750}

and typical classifications:


<img src="https://github.com/theScinder/digitalDigitRecognition/blob/master/zeroTo11Results_1/intModelDemo0to11_1.png">
<img src="https://github.com/theScinder/digitalDigitRecognition/blob/master/zeroTo11Results_1/intModelDemo0to11_2.png">

It is easy to notice that the lighting is worse on the 0-10 dataset, and the placement of gestures in the images is also more likely to go partially outside the frame. The images per category are also lower - the total dataset was only slightly larger (690 vs. 640) than the 0-3 dataset before mirroring the images. QED the results should improve with a larger, higher quality dataset. 
