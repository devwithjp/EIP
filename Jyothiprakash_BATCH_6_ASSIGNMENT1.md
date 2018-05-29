
## Convolution

In Convolution Neural Network the network utilizes a mathematical operation called convolution instead of general multiplication in at least one of the layers.
Convulution is a specialized linear operation, which operates on two real valued functions .

Mathematically, convolution can be better understood with the below example:

Say we want to track the speed of a cricket ball with a sensor which gives the output x(t),_position_ of ball at time _t_. Suppose the laser's output is noisy, then we'd want average of several measurements. And relevancy of measurements(latest measurements may be more relevant) vary, thus we would require weighted average.

Weighted average operation at every moment is termed as convolution and it is given by:

$$  y(t) - \int x(a)w(t-a)da $$

$$ y(t) - (x*w)(t) $$ 
where * denotes convolution.

This approach is continuous, but since in in most cases discrete convolution is used. For example say that the positioin of the ball is taken at every 0.0001 s, then
$$ y(t) - \sum_{-\infty}^{\infty}  x(a)w(t-a) $$

Where, w is a probability density function which takes the weights into consideration.

As per convolution network terminologies, the first argument(function x) is referred to as the **_input_** and the second argument( function w) is referred as the **_kernel_**. And the output of convolution is referred to as the **_feature map_**. Weights of *filters* are *kernels*

Convolution is commutative, utilizing this the kernel and input can be flipped in the function. Thus Cross- correlation is  same as convolution, only difference is that there is no flipping the kernel in cross-correlation. Usually cross- correlation is referred to as convolution in ML context.

In Machine learning,The input is a multidimensional array of data and the kernel are multidimensional array of parameters which are the weights. Thus often more than one axis is used at a time i.e if a 2-d image is used as input then 2-d kernel is also used.

3x3 Convolution is simply where the kernel is of the dimensions 3x3. It is a feature extractor.
1x1 Convolution has a kernel of dimensions 1x1. 

Example for convolution using 5x5x3 kernel is given below:-
 
 ![Convolving an image with a filter](https://cdn-images-1.medium.com/max/800/1*FUEkm0JghT3ab8P7p9c5Qg.png)
A 5x5x3 filter is used and it is traversed through the whole image convolving along the way.

![Upon convolution over a region](https://cdn-images-1.medium.com/max/800/1*3sfzVenrdS5MWGsmSCbx3A.png)
For every convolution at any instant yields one scalar number.  

![Result!](https://cdn-images-1.medium.com/max/800/1*mcBbGiV8ne9NhF3SlpjAsA.png)

The result is a layer that is of dimensions= (32-5)x(32-5)x3= 28 x 28 x 3


###### References: 
[Ian goodfellow:Deeplearning](http://www.deeplearningbook.org/contents/convnets.html)
[Wikipedia-Discrete Convolution](https://en.wikipedia.org/wiki/Convolution#Discrete_convolution+)
[CNN-Medium](https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8)

---------------------------------------------

## Feature and Feature Engineering 

A feature is an _individually measurable_ property or _characteristic_ of the phenemenon.
For example,
- In computer vision,features may be structures in the images like points, edges or objects
- In speech recognition, features may be length of sounds, relative powers etc

Feature engineering is the transformation of raw data to features that is suiatble for modelling. It is the process of using knowledge of the data to create features that is suitable for the ML algorithm to work.  
Upon proper execution of feature engineering, predictive power of the learning algorithm can be increased.
Feature engineering is more like an art which determines if a model is good or bad.Thus it is vital to success to ML.

Steps done in Feature engineering are

1. Brainstorming Features: Figuring out and selecting useful features from the data
2. Extracting  those features
3. Creating features: Construction of new features based on the data provided
4. Checking how those features work with the model
5. Improving features if neccessary
6. Start again from begining untill features which work perfectly are found.



###### References: 
[Wikipedia-Feature (machine learning)](https://en.wikipedia.org/wiki/Feature_(machine_learning))
[Wikipedia-Feature Engineering](https://en.wikipedia.org/wiki/Feature_engineering)
[Medium-What Is Feature Engineering for Machine Learning?](https://medium.com/mindorks/what-is-feature-engineering-for-machine-learning-d8ba3158d97a)

------------------------------------------------


### Activation Function

Everyday our brain encounters a lot of information. It is constantly working on to segregate and seperate usefull information from useless information. This functionality is also neccessary in case of neural network. Activation function helps the network to carry out that functionality. They help the network use the useful information while supressing irrelevant data. It basically decides whether a neuron(basic unit of a neural network) must be activated or not. In other words, the function decide whether the information that the neuron is recieving is relavant for the given information or if it should be ignored.  
Feature map upon operation by some activating function gives a neuron.
Activation function is the non linear transformation applied over the the input signal where the output is then sent to the next layer as the input.
A Neural Net without an activation function is simply a Linear regression model. Linear Regression model have limited power, capability and poor performance. It can't learn complicated/non-linear-big data like images,videos,audio,speech etc without activation functions

Some popular activation functions are:

1. Sigmoid or Logistic

  $$ f(x) = 1 / 1 + exp(-x) $$
   
   It's Range is between 0 and 1.
 
  ![Sigmoid function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/600px-Logistic-curve.svg.png)
 
2. Tanh - Hyperbolic tangent
 $$  f(x) = 1-exp(-2x)/1+exp(-2x) $$
 ![Tan Hyperbollic](http://math.feld.cvut.cz/mt/txtb/4/gifa4/pe3ba4fg.gif)
 
3. ReLu -Rectified linear units
 Better than Tanh by 6 times in terms of convergence

 Defined as,
 $$ R(x) = max(0,x) $$
 $$ if x < 0 $$ 
 $$  R(x) = 0 $$
 $$ if x >= 0 $$
 $$ R(x) = x $$
  ![ReLu](https://i.imgur.com/gKA4kA9.jpg)
  
 Its limitation is that it should only be used within Hidden layers    of a Neural Network Model.
 Also it can result in dead neurons.

Identity function is used as an activation function when no operation over the input is neccessary.

###### References:
[Analytics vidhya-Fundamentals of Deep Learning – Activation Functions and When to Use Them](https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/)
[Medium-Activation functions and it’s types](https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f)

-----------------------------

### How to create an account on GitHub and upload a sample project

#### How to create an account on GitHub

1. Go to [GitHub](https://github.com/) and sign up
2. Select the _Unlimited public repositories for free_ option.
3. Verify from the verification email that's sent to the registration E-mail.
4. Click the E-mailed link to complete the verification process.

#### How to upload a sample project on GitHub

1. Create a new repository or go to an existing repository
   click + on the upper right corner and then click new repository. Provide name and description and choose if you want to keep it private(hidden) or public.Initialize with a readme and Finally create repo.

2. Under your repo name, click _upload files-
3. Drag and drop the file you wish to upload onto the file  tree
4. Provide description pertaining to the changes made to the file at the bottom of the page(commit message).
5. Select whether to add the commit to the current branch or to another branch. If the current branch is master, then you have to choose to create a new branch and create a pull reuest.
6. Click commit changes
   
-----------------------------------

### Receptive field

Biologically, receptive field of a neuron is the region of sensory space that is responsible for firing of the neuron.
In CNN when dealing with complex data like images, it's not sensible to connect neurons to all neurons. Instead, we connect each neuron to a local region of the input. The spatial extent of this connectivity is call as the receptive field.Basically, receptive field is the region in the input space that a particular CNN feature is looking at or getting impacted by.
Receptive field is divided into local receptive and globle receptive field.
Local receptive field is the region of space from the previous layer that's recieved by the current layer.
Globle recieptive field is the region of space from all the previous layers that's recepted by the current layer.Increasing number of layers, increases the globle receptive field.
Finally, if the globle receptive field is equal to the whole input image then proper number of layers were chosen.

###### References:

[Blog: Christian S Perone-The effective receptive field on CNNs](http://blog.christianperone.com/2017/11/the-effective-receptive-field-on-cnns/)
[Medium-A guide to receptive field arithmetic for Convolutional Neural Networks](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)

-------------------------------------
### 10 examples of use of MathJax in Markdown
All the equations in mathjax form are enclosed between $\$<formula> \$$ to display an equation
1. Gravity equation
  MathJax: F= {{Gm_1m_2 }\over r^2}
  Equation after enclosing between \$\$..\$\$ :
  $$ F= {{Gm_1m_2 }\over r^2}$$
2. Sum to n numbers
  MJ  : \sum_{i=0}^{n}a_i
  Equation after enclosing between \$\$..\$\$ :
 $$\sum_{i=0}^{n}a_i$$
3. Quadratic formula
   MJ : x = {-b \pm \sqrt{b^2-4ac} \over 2a}
    Equation after enclosing between \$\$..\$\$:
 $$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$
4. Pythagorean Theorem
  MJ: a^2+b^2=c^2
  $$ a^2+b^2=c^2 $$
5. Binomial probability formula
  MJ:P(X)=C_x^np^xq^{n-x}
  $$P(X)=C_x^np^xq^{n-x} $$
6. Linear Regression formula
  
   MJ: y=a+bx
   b(slope)={{n∑xy-(∑x)(∑y)} \over {n∑x^2-      (∑x)^2}
   a(intercept)={{n∑y-b(∑x)}\over{n}}
 
  $$y=a+bx$$
  $$b(slope)={{n∑xy-(∑x)(∑y)} \over {n∑x^2-(∑x)^2}}$$
  $$a(intercept)={{n∑y-b(∑x)}\over{n}}$$
  
7. Taylor Series
 MJ:  \sum_{n=0}^\infty {f^n (a) \over n!}{(x-a)^n}
  $$\sum_{n=0}^\infty {f^n (a) \over n!}{(x-a)^n}$$
8. Law of sines
  MJ:{a \over SinA} = {b \over SinB}={c \over SinC}
  $$ {a \over SinA} = {b \over SinB}={c \over SinC} $$
9. Laws of Exponents
    
     MJ:(a^m)(a^n) = a^{m+n} 
     (ab)^m = a^ m b^m
     (a^m)^n = a^{mn}
     
  $$ (a^m)(a^n) = a^{m+n} $$
   $$(ab)^m = a^ m b^m $$
   $$(a^m)^n = a^{mn}$$

10. Permutation
   MJ:P_r^n={n!\over (n-r)!}
     
   $$P_r^n={n!\over (n-r)!}$$
   ----------------------------------------------------------------
   ####   Jyothiprakash S 
  ####   Batch 6
  ##### Assignment 1
   
   
   