# ECBM-E6040-HOMEWORK-2-solution

Download Here: [ECBM E6040 HOMEWORK #2 solution](https://jarviscodinghub.com/assignment/ecbm-e6040-homework-2-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

INSTRUCTIONS: This homework contains two parts – theoretical and programming. Submission for this homework will be via bitbucket repositories created for each student and should contain the following
1. A ﬁle called hw2 writeup.pdf that contains solutions to the theoretical questions
2. Put all ﬁgures and discussions, and document all parameters you used in the programming question in the IPython notebook ﬁle, hw2b.ipynb, which is already included in the homework 2 repository. All the discussions should also be included in the notebook ﬁle.
Please be advised that the programming part of this homework may take some time to ﬁnish, so start early!
Theoretical
You will need the deﬁnition of the PDF of a matrix normal distribution to complete this part. The following provides a PDF of a simpliﬁed special case that will be used in this assignment. Matrix Normal Distribution: A n×n square matrix valued random variable A is said to follow a matrix normal distribution (MN) with parameters (M,λ−1 2I,λ−1 2I) if it has the following probability density function P?A;(M,λ−1 2I,λ−1 2I)?= 1 (2π) n2 2 exp?−1 2 Tr?λ(A−M)T(A−M)??
PROBLEM a (40 points) Given the observed data (x1,y1),··· ,(xm,ym), where xi ∈Rn and yi ∈Rn, ∀ i ∈ [1,m], we are interested in ﬁnding an A ∈ Rn×n such that in some sense (to be deﬁned below) yi ≈ Axi.
For simplicity, we use the following notation Y =?y1 y2 ··· ym? X =?x1 x2 ··· xm? and assume that m n, and that XXT is invertible.
(i) For the least squares loss function
Lls =
m X i=1?yi −Axi?T?yi −Axi? ﬁnd Als = argmin A Lls. (ii) For the least squares loss function with Frobenius norm regularization term
Lr = λkAk2 F +
m X i=1?yi −Axi?T?yi −Axi?
ﬁnd Ar = argmin A Lr. Note: kAk2 F = Tr(ATA). (iii) Assume that the errors ?i = yi − Axi are normally distributed with mean 0 under the ideal A, i.e., ?i ∼N(0,σ2I). Find AML, the maximum likelihood estimate of A.
(iv) Under the same assumption of normal error distribution, consider a prior on A of the form A ∼ MN(M,λ−1 2,I,λ−1 2I). Find AMAP, the maximum a posteriori estimate of A. What will be AMAP if we assume M to be the zero matrix.
(v) Comment on the relation between the expressions derived in (i) and (iii), and of those derived in (ii) and (iv).
2
Programming
For this part, you will experiment with diﬀerent Multilayer Perceptron conﬁgurations, and empirically study various relationships among number of layers and number of parameters. You should start by going through the Deep Learning Tutorials Project. In particular, the source code provided in the Homework 2 repository is excerpted from logistic sgd.py and mlp.py.
You are asked to partially reproduce the phenomena shown in Figure 6.9 and Figure 6.10 of the textbook. The original work for these two ﬁgures implemented an advanced framework of deep network [2], which is beyond the material covered in the course till now. Instead of reimplementing the original work, you should simply use the multilayer perceptron described in the tutorial.
You will be using the street view house numbers (SVHN) dataset [1]. The dataset is similar in ﬂavor to MNIST, but contains substantially more labeled data, and comes from a signiﬁcantly harder, real world problem (recognizing digits and numbers in natural scene images). You will use the Format 2 of the SVHN dataset. Each sample in this format is a MNIST-like 32-by-32 RGB image centered around a single character. Many of the images do contain some distractors on the sides, which of course makes the problem interesting.
The task is to implement an MLP to classify the images of the SVHN dataset. The input to the MLP is a color image, and the output is a digit between 0 and 9.
A python routine called load data is provided to you for downloading and preprocessing the dataset. You should use it, unless you have absolute reason not to. The ﬁrst time you call load data, it will take some time to download the dataset (about 180 MB). Please be careful NOT TO commit the dataset ﬁles into the repository.
Note that all the results, ﬁgures, and parameters should be placed inside the IPython notebook ﬁle hw2b.ipynb.
PROBLEM b (60 points)
1. First enable the construction of MLPs with multiple hidden layers. Implement your MLP in the skeleton myMLP class in hw2b.py.
2. Implement an MLP with 2 hidden layers. Compare the eﬀect of two activation functions, tanh and softmax, on neurons in hidden layers with other parameters ﬁxed. Note that the output neuron always uses softmax. Document your choice of parameters explicitly, and discuss your test accuracy results in both cases.
3. Experiment with the number of hidden layers. In particular, generate a plot similar to the one in Figure.1. Note that Figure.1 is similar in spirit to Figure 6.9 of the textbook. Each hidden layer should contain the same amount of
3
neurons. You might want to start with a network with 3 hidden layers, and experiment with parameters (e.g., activation function, learning rate, number of hidden neurons, etc.). After ﬁnding a set of parameters, run your MLP 8 times with the number of hidden layers varying from 1 to 8 (the total number of layers thus ranges from 3 to 10). Document your choice of parameters explicitly, and discuss your test accuracy results.
2 4 6 8 10 12 14 Number of Layers
82
84
86
88
90
92
94
96
Test Accuracy, [%]
MLP on SVHN Dataset
Figure 1:
4. Experiment with the number of hidden layers, but ﬁx the total number of neurons in all hidden layers. In particular, generate a plot similar to Figure.2. In Figure.2, the total amount of hidden neurons is ﬁxed at 2.4K. You may chose another number. Each hidden layer should contain the same amount of neurons (that is,b total number number of layerscneurons in each layer). Run your MLP 8 times with number of hidden layers varying from 1 to 8. Document your choice of parameters and their number explicitly, and discuss your test accuracy results.
4
0 1 2 3 4 5 6 7 8 Number of Hidden Layers
82
84
86
88
90
92
94
96
Test Accuracy, [%]
Effect of Number of Layers Total Number of Parameters in Hidden Layers is fixed to 2.4K
Figure 2:
5. For a ﬁxed number of hidden layers experiment with the number of neurons in hidden layers. In particular, generate a plot similar to the one in Figure.3. Note that Figure.3 is similar in spirit to Figure 6.10 in the textbook. Run your MLP with 1 hidden layer 5 times with 5 diﬀerent numbers of hidden neurons. Repeat the above experiment with 2 hidden layers. Document your choice of parameters and their number explicitly, and discuss your test accuracy results.
5
0 500 1000 1500 2000 2500 Number of Neurons in the Hidden Layer
81.0
81.5
82.0
82.5
83.0
83.5
84.0
Test Accuracy, [%]
Effect of Number of Neurons in Hidden Layer The MLP contains 1 hidden layer
Figure 3:
NEED HELP:
If you have any questions you are advised to use Piazza forum which is accessible through courseworks.
GOOD LUCK!
References
[1] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng, “Reading Digits in Natural Images with Unsupervised Feature Learning,” NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011.
[2] Ian Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, Vinay Shet, “Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks,” ICLR 2014.
