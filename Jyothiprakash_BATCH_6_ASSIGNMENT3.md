
-----------------------------
### Dilated Convolution 

Regular convoltutions have filters which are continuous. In Dilated convolutions the filters aren't continuos instead contain spaces between each cell, termed dilations.

For example, assume a filter/kernel of size 3x3 would compute over 9 cells at a time. However, in  dilated convolution, a kernel size of 3x3 consisting of 9 weights could be used to convolve over 25 cells(5x5) or more at a time, consisting of one space between each cell. In other words, receptive field of the kernel is enhanced as it looks over more cells than before, but contains spaces in between. Hence, diluted convolution's kernel slides over the input(image) faster than usual, thus effective receptive field would grow much quicker in dilated convolutions. Also, this process enables a wider field of view at the same computational cost. 

![Dilated convolution of dilation factor=2](https://i.stack.imgur.com/qA0Kx.gif)

The usual convolution can be thought of as a dilated convolution  with 0 dilation (0 spaces), (dilation rate/factor =1). A 3x3 kernel with dilation factor equal to 2 would behave as a 5x5 kernel, and so on.


Dilated convolution finds profound uses in areas which require image segmentation. This is because dilated convolution filter has a broader view of the input hence learn to seperate the subject from the backgroung and so on. Some other applications are semantic segmentations,super resolutions, denoising, key-point detection etc.

References:
[cs231n-ConvNets](http://cs231n.github.io/convolutional-networks/)
[Towards data science-Understanding dilated convolution](https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-tensorflow-with-d376b3972b25)
[An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)

--------------------------------
### Depthwise Seperable Convolution

In depthwise seperable convolution 2 major steps are involved,
1. Depthwise convolution- Convolution is performed **independently** over each channel of input.
2. Poitwise coonvolution- A 1x1 convolution is performed to merge the outputs of the above step, projecting the channels output obtained by the depthwise convolution onto another new channel

For example, consider a 3x3 covolution on a 16 input channel and 32 output channels. 
In usual convolution, every channel of the 16 channels is traversed by 32 3x3 kernels resulting 512(16x32) feature maps, then a feature map out of every channel is merged by adding them up to obtain 32 channels that's required.
In contrast, in depthwise seperable convolution, we traverse the 16 channels using 1 3x3 kernel each, obtaining 16 feature maps, then these 16 feature maps are convolved with 32 1x1 convolutions each and then added together. This process utilizes 656(16x3x3 = 16x32x1x1) computations as opposed to the 4608(16x32x3x3) utilized in normal convolution.

In conclusion, depthwise seperable convolution are prominent in DNN because of the following advantages,

1. The process involves lesser parameters than usual convolution layers. Also, due to this they're less prone to overfitting
2. Since they require fewer parameters, they require lesser computations, thus are faster and cheaper.





References:
[Towards Data Science-An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
 [Illarion Khlestov Blog-Convolutions Types](https://ikhlestov.github.io/pages/machine-learning/convolutions-types/)
 [Eli Bendersky's website-Depthwise separable convolutions for machine learning](https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/)
 
 ---------------
 
### Transposed Convolution
#### Also called Deconvolution/Fractionally strided convolution
 
 In transpose convolution, input is padded and spaces are added in between so that after covolvution the output is larger than input dimensions. This isn't exactly a mathematical inverse of convolution. 
 Decovolution layer upsamples the input to get the image before convolution. Deconvolution is a trainable upsampling convolutional layer, whose parameters are updated while training. 
 
 A deconvolution operation is performed in the same way as a normal convolution. Just have to insert zeros(spaces) betweeb the cosecutive inputs.

 ![Deconvolution without any padding or strides](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/no_padding_no_strides_transposed.gif)
 ![Deconvolution with padding and strides](https://github.com/vdumoulin/conv_arithmetic/raw/master/gif/padding_strides_transposed.gif)
 
Transposed convolution are used in applications such as resolutioin enhancement, noise removal, colorization etc.
 
In conclusion, deconvolutioin is a transformation that goes in opposite direction of normal convolution. In conclusion, output of convolution becomes the input of deconvolution and input of convolution is the output of deconvolution.

References:
[Towards data science-Up-sampling with Transposed Convolution](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0)
[Towards data science-An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
[CV tricks-Deconvolution](http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/)

-------------------------------- 
--------------
#### Assignment 3
##### Jyothiprakash S
##### Batch 6

-----------------------------------