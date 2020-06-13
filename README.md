# 101 Machine Learning Algorithms

At Data Science Dojo, our mission is to make data science (machine learning in this case) available to everyone. Whether you join our [data science bootcamp](https://datasciencedojo.com/data-science-bootcamp/), read our blog, or [watch our tutorials](https://tutorials.datasciencedojo.com/), we want everyone to have the opportunity to learn data science.

Having said that, each accordion dropdown is embeddable if you want to take them with you. All you have to do is click the little 'embed' button in the lower left-hand corner and copy/paste the iframe. All we ask is you link back to this post.

By the way, if you have trouble with Medium/TDS, just throw your browser into incognito mode.

# classificationalgorithms(18)

Any of these classification algorithms can be used to build a model that predicts the outcome class for a given dataset. The datasets can come from a variety of domains. Depending upon the dimensionality of the dataset, the attribute types, sparsity, and missing values, etc., one algorithm might give better predictive accuracy than most others. Let’s briefly discuss these algorithms.

## Decision Tree

A decision tree classification algorithm uses a training dataset to stratify or segment the predictor space into multiple regions. Each such region has only a subset of the training dataset. To predict the outcome for a given (test) observation, first, we determine which of these regions it belongs to. Once its region is identified, its outcome class is predicted as being the same as the mode (say, ‘most common’) of the outcome classes of all the training observations that are included in that region. The rules used to stratify the predictor space can be graphically described in a tree-like flow-chart, hence the name of the algorithm. The only difference being that these decision trees are drawn upside down.Decision tree classification models can easily handle qualitative predictors without the need to create dummy variables. Missing values are not a problem either. Interestingly, decision tree algorithms are used for regression models as well. The same library that you would use to build a classification model, can also be used to build a regression model after changing some of the parameters.          Although the decision tree-based classification models are very easy to interpret, they are not robust.  One major problem with decision trees is their high variance. One small change in the training dataset can give an entirely different decision trees model. Another issue is that their predictive accuracy is generally lower than some other classification models, such as “Random Forest” models (for which decision trees are the building blocks).

![Fig. 1: Decision Tree Example](https://datasciencedojo.com/wp-content/uploads/decision-tree-example.png)

- [R Tutorial](https://blog.datasciencedojo.com/classification-decision-trees/)

## Decision Stump

A Decision Stump is a decision tree of 1 level. They are also called 1-rules and use one feature to arrive to a decision. Independently, a Decision Stump is a 'weak' learner, but they can be effective when used as one of the models in bagging and boosting techniques, like AdaBoost.If the data is discrete it can be divided in terms of frequency and continuous data can be divided by a threshold value. The graph on the left-hand side of this image shows a dataset divided linearly by a decision stump.

![Fig. 2: A dataset divided linearly by a decision stump.](https://i.stack.imgur.com/brE2F.png)

- [R Example](https://www.r-bloggers.com/the-power-of-decision-stumps/)

## Naive Bayes

Naive Bayes Classifier is based on the Bayes Theorem. The Bayes Theorem says the conditional probability of an outcome can be computed using the conditional probability of the cause of the outcome.The probability of an event x occurring, given that event C has occurred in the prior probability. It is the knowledge that something has already happened. Using the prior probability, we can compute the posterior probability - which is the probability that event C will occur given that x has occurred. The Naive Bayes classifier uses the input variable to choose the class with the highest posterior probability.The algorithm is called naive because it makes an assumption about the distribution of the data. The distribution can be Gaussian, Bernoulli or Multinomial. Another drawback of Naive Bayes is that continuous features have to be preprocessed and discretized by binning, which can discard useful information.

- [Tutorial](https://blog.datasciencedojo.com/unfolding-naive-bayes-from-scratch-part-1/)
- [R Example](https://rpubs.com/riazakhan94/naive_bayes_classifier_e1071)

## Gaussian Naive Bayes

The Gaussian Naive Bayes algorithm assumes that all the features have a Gaussian (Normal / Bell Curve) distribution. This is suited for continuous data e.g Daily Temperature, Height. The Gaussian distribution has 68% of the data in 1 standard deviation of the mean, and 96% within 2 standard deviations. Data that is not normally distributed produces low accuracy when used in a Gaussian Naive Bayes classifier, and a Naive Bayes classifier with a different distribution can be used.

- [Python Example](https://www.antoniomallia.it/lets-implement-a-gaussian-naive-bayes-classifier-in-python.html)

## Bernoulli Naive Bayes

The Bernoulli Distribution is used for binary variables - variables which can have 1 of 2 values. It denotes the probability of of each of the variables occurring. A Bernoulli Naive Bayes classifier is appropriate for binary variables, like Gender or Deceased.

- [Python Example](https://chrisalbon.com/machine_learning/naive_bayes/bernoulli_naive_bayes_classifier/)

## Multinomial Naive Bayes

The Multinomial Naive Bayes uses the multinomial distribution, which is the generalization of the binomial distribution. In other words, the multinomial distribution models the probability of rolling a k sided die n times.Multinomial Naive Bayes is used frequently in text analytics because it has a bag of words assumption - which is the position of the words doesn't matter. It also has an independence assumption - that the features are all independent.

- [Python Example](https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67)

## K Nearest Neighbours (KNN)

K Nearest Neighbors is a the simplest machine learning algorithm. The idea is to memorize the entire dataset and classify a point based on the class of its K nearest neighbors.Figure 3 from Understanding Machine Learning, by Shai Shalev-Shwartz and Shai Ben-David, shows the boundaries in which a label point will be predicted to have the same class as the point already in the boundary. This is a 1 Nearest Neighbor, the class of only 1 nearest neighbor is used.KNN is simple and without any assumptions, but the drawback of the algorithm is that it is slow and can become weak as the number of features increase. It is also difficult to determine the optimal value of K - which is the number of neighbors used.

- [R Example](https://www.datatechnotes.com/2018/10/learning-vector-quantization.html)

## Support Vector Machine (SVM)

An SVM is a classification and regression algorithm. It works by identifying a hyper plane which separates the classes in the data. A hyper plane is a geometric entity which has a dimension of 1 less than it's surrounding (ambient) space.If an SVM is asked to classify a two-dimensional dataset, it will do it with a one-dimensional hyper place (a line), classes in 3D data will be separated by a 2D plane and Nth dimensional data will be separated by a N-1 dimension line.SVM is also called a margin classifier because it draws a margin between classes. The image, shown here, has a class which is linearly separable. However, sometime classes cannot be separated by a straight line in the present dimension. An SVM is capable of mapping the data in higher dimension such that it becomes separable by a margin.Support Vector machines are powerful in situations where the number of features (columns) is more than the number of samples (rows). It is also effective in high dimensions (such as images). It is also memory efficient because it uses a subset of the dataset to learn support vectors.

![Fig. 4: Margin Classifier](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_71.png)

- [Python Example](https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/)

## Linear Support Vector Classifier (SVC)

A Linear SVC uses a boudary of  one-degree (linear / straight line) to classify data. It has much less complexity than a non-linear classifier and is only appropriate for small datasets. More complex datasets will require a non linear classifier.

- [Python Example](https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/)

## NuSVC

NuSVC uses Nu parameters which is for regularization. Nu is the upper bound on the expected classification error. If the value of Nu us 10% then 10% of the data will be misclassified.

- [Python Example](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)

## Stochastic Gradient Descent (SGD) Classifier 

SGD is a linear classifier which computes the minima of the cost function by computing the gradient at each iteration and updating the model with a decreasing rate. It is an umbrella term for many types of classifiers, such as Logistic Regression or SVM) that use the SGD technique for optimization.

- [R Example](https://rpubs.com/aaronsc32/quadratic-discriminant-analysis)
- [Python Example](https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/)

## Bayesian Network

A Bayesian Network is a graphical model such that there are no cycles in the graph. This algorithm can model events which are consequences of each other. An event that causes another points to it in the graph. The edges of the graph show condition dependence and the nodes are random variables.

- [R Tutorial](https://www.r-bloggers.com/bayesian-network-in-r-introduction/)
- [Graph Source](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/glossary/bayesnet.html)

## Logistic Regression

Logistic regression estimates the relationship between a dependent categorical variable and independent variables. For instance, to predict whether an email is spam (1) or (0) or whether the tumor is malignant (1) or not (0).If we use linear regression for this problem, there is a need to set up a threshold for classification which generates inaccurate results. Besides this, linear regression is unbounded, and hence we dive into the idea of logistic regression. Unlike linear regression, logistic regression is estimated using the Maximum Likelihood Estimation (MLE) approach. MLE is a "likelihood" maximization method, while OLS is a distance-minimizing approximation method. Maximizing the likelihood function determines the mean and variance parameters that are most likely to produce the observed data. Logistic Regression transforms it's output using the sigmoid function in the case of binary logistic regression. As you can see in Fig. 5, if ‘t’ goes to infinity, Y (predicted) will become 1 and if ‘t’ goes to negative infinity, Y(predicted) will become 0.The output from the function is the estimated probability. This is used to infer how confident can predicted value be as compared to the actual value when given an input X. There are several types of logistic regression:

![Fig. 5: Sigmoid Function](https://datasciencedojo.com/wp-content/uploads/Logistic-Regression-Sigmoid-function.png)

- [Python Example](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)

## Zero Rule (ZeroR)

ZeroR is a basic classification model which relies on the target and ignores all predictors. It simply predicts the majority category (class). Although there is no predictibility power in ZeroR, it is useful for determining a baseline performance as a benchmark for other classification methods. This is the least accurate classifier that we can have. For instance, if we build a model whose accuracy is less than the ZeroR model then it's useless.The way this algorithm works is that it constructs a frequency table for the target class and select the most frequent value as it's predicted value regardless of the input features.

- [Python Example](https://machinelearningmastery.com/implement-baseline-machine-learning-algorithms-scratch-python/)

## One Rule (OneR)

This algorithm is also based on the frequency table and chooses one predictor that is used for classification.It generates one rule for each predictor in the data set, then selects the rule with the smallest total error as its "One Rule". To create a rule for the predictor, a frequency table is constructed for each predictor against the target.

- [R Example](https://christophm.github.io/interpretable-ml-book/rules.html)

## Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA) is performed by starting with 2 classes and generalizing to more. The idea is to find a direction, defined by a vector, such that when the two classes are projected on the vector, they are as spread out as possible.

- [Python Example](https://sebastianraschka.com/Articles/2014_python_lda.html)[Vector Source: Ethem Alpaydin](https://www.cmpe.boun.edu.tr/~ethem/i2ml/i2ml-figs.pdf)

## Quadratic Discriminant Analysis (QDA)

QDA is the same concept as LDA, the only difference is that we do not assume the distribution within the classes are normal. Therefore, a different covariance matrix has to be built for each class which increases the computational cost because there are more parameters to estimate, but it fits data better than LDA.

- [R Example](https://rpubs.com/aaronsc32/quadratic-discriminant-analysis)

## Fisher's Linear Discriminant

Fisher's Linear Discriminant improves upon LDA by maximizing the ratio between class variance and the inter class variance. This reduces the loss of information caused by overlapping classes in LDA. 	

# regressionanalysis (20)

Regression Analysis is a statistical method for examining the relationship between two or more variables. There are many different types of Regression analysis, of which a few algorithms can be found below.

## Linear Regression

Linear Regression models describe the relationship between a set of variables and a real value outcome. For example, input of the mileage, engine size, and the number of cylinders of a car can be used to predict the price of the car using a regression model.Regression differs from classification in how it's error is defined. In classification, the predicted class is not the class in which the model is making an error. In regression, for example, if the actual price of a car is 5000 and we have two models which predict the price to be 4500 and 6000, then we would prefer the former because it is less erroneous than 6,000. We need to define a loss function for the model, such as Least Squares or Absolute Value.The drawback of regression is that it assumes that a single straight line is appropriate as a summary of the data.

- [R Example](http://r-statistics.co/Linear-Regression.html)

## Polynomial Regression

Polynomial Regression is the same concept as linear regression except that it uses a curved line instead of a straight line (which is used by linear regression). Polynomial regression learns more parameters to draw a non-linear regression line. It is beneficial for data that cannot be summarized by a straight line.The number of parameters (also called degrees) has to be determined. A higher degree model is more complex but can over fit the data.

- [Python Example](https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/)

## Poisson Regression

Poisson Regression assumes that the predicted variables follows a Poisson Distribution. Hence, the values of the predicted variable are positive integers. The Poisson distribution assumes that the count of larger numbers is rare and smaller values are more frequent. Poisson regression is used for modelling rare occurrence events and count variables, such as incidents of cancer in a demographic or the number of times power shuts down at NASA.

- [R Example](https://www.r-bloggers.com/generalized-linear-models-poisson-regression/)

## Ordinary Least Squares (OLS) Regression

Least Squares is a special type of Regression model which uses squares of the error terms as a measure of how accurate the model is.Least Squares Regression uses a squared loss. It computes the difference between the predicted and the actual value, squares it, and repeats this step for all data points. A sum of the all the errors is computed. This sum is the overall representation of how accurate the model is.Next, the parameters of the model are tweaked such that this squared error is minimized so that there can be no improvement.For this model, it is appropriate to preprocess the data to remove any outliers, and only one of a set of variables which are highly correlated to each other should be used.

- [R Example](https://www.r-bloggers.com/ordinary-least-squares-ols-linear-regression-in-r/)

## Ordinal Regression

Also called ranking learning, ordinal regression takes a set of ordinal values as input. Ordinal variables are on an arbitrary scale and the useful information is their relative ordering. For example, ordinal regression can be used to predict the rating of a musical on a scale of 1 to 5 using ratings provided by surveys. Ordinal Regression is frequently used in social science because surveys ask participants to rank an entity on a scale.

- [R Example](https://www.r-bloggers.com/how-to-perform-ordinal-logistic-regression-in-r/)

## Support Vector Regression

Support Vector Regression works on the same principle as Support Vector Machine except the output is a number instead of a class. It is computationally cheaper, with a complexity of O^2*K where K is the number of support vectors, than logistic regression.

- [R Example](https://www.kdnuggets.com/2017/03/building-regression-models-support-vector-regression.html)

## Gradient Descent Regression

*Gradient Descent Regression uses gradient descent to optimize the model (as opposed to, for example, Ordinary Least Squares). Gradient Descent is an algorithm to reduce the cost function by finding the gradient of the cost at every iteration of the algorithm using the entire dataset.

- [R Example](https://www.r-bloggers.com/linear-regression-by-gradient-descent/)

## Stepwise Regression

Stepwise regression solves the problem of determining the variables, from the available variables, that should be used in a regression model. It uses F-tests and t-tests to determine the importance of a variable. R-squared, which explains the ratio of the predicted variable explained by a variable, is also used.Stepwise regression can either incrementally add and/or remove a variable from the entire dataset to the model such that the cost function is reduced.

- [R Example](http://r-statistics.co/Model-Selection-in-R.html)

## Lasso Regression (Least absoulute selection and shrinkage operator)

Often times, the data we need to model demands a more complex representation which is not easy to characterize with the simple OLS regression model. Hence, to produce a more accurate representation of the data, we can add a penalty term to the OLS equation. This method is also known as L1 regularization.The penalty term imposes a constraint on the total sum of the absolute values of the model parameters. The goal of the model is to minimize the error represented in Fig. 6 which is the same as minimizing the SSE with an additional constraint. If your linear model contains many predictor variables or if these variables are correlated, the traditional OLS parameter estimates have large variance, thus making the model unreliable. This leads to an over-fitted model. A penalty term causes the regression coefficients for these unimportant variables to shrink towards zero. This process allows the model to identify the variables strongly associated with the output variable, thereby reducing the variance.Lambda, a tuning parameter, is used to control the strength of the model penalty in Lasso Regression. As lambda increases, more coefficients are reduced to zero. This feature selection process can help alleviate multi-collinearity because Lasso tends to select only one of the correlated features and shrink the other to zero. Lasso is generally used when we have a greater number of features, because it automatically performs feature selection.

![Fig 6: Lasso Regression Loss Function](https://datasciencedojo.com/wp-content/uploads/lasso.png)

- [R Example](http://ricardoscr.github.io/how-to-use-ridge-and-lasso-in-r.html)

## Ridge Regression (L2)

Ridge regression uses ridge regularization to prepare a regression model. Ridge regularization adds the square of the coefficients to the cost function. It is effective if there are multiple coefficients with large values. It makes the values of the coefficients of the indiscriminate variables small.

- [R Example](https://drsimonj.svbtle.com/ridge-regression-with-glmnet)

## Elastic Net Regression

Elastic Net generalizes the idea of both Ridge and Lasso regression since it combines the penalties from both L1 ( Lasso) and L2 (Ridge) regularization. Elastic Net aims at minimizing the loss function represented in Fig. 7. 𝞪 is the tuning parameter which can be changed to implement both Ridge and Lasso regression alternatively or simultaneously to optimize the elastic net. If you plug in 𝞪 = 0, the penalty function corresponds to ridge and 𝞪 = 1 corresponds to Lasso regularization.In the case of correlated independent variables in a dataset, the Elastic Net will group these variables together. Now if any one of the variable of this group is strongly associated with the dependent variable, then the entire group will be a part of the model, because selecting only one of those variables (like what we did in Lasso) might result in losing some useful information, leading to a poor model performance. Hence, elastic net produces grouping in case of multi-collinearity.The size of the respective penalty terms Lambda and alpha can be tuned via cross-validation to find the model's best fit.

![Fig. 7: Elastic Net Loss Function](https://datasciencedojo.com/wp-content/uploads/elastic-net-loss-function.png)[R Example](https://daviddalpiaz.github.io/r4sl/elastic-net.html)

## Bayesian Linear Regression

In the Bayesian world, linear regression is formulated using probability distributions rather than point estimates. The dependent variable, Y, is not estimated as a single value, but is assumed to be drawn from a probability distribution. Y is generated from a normal distribution with a mean and variance. Bayesian Linear Regression aims to find the posterior distribution for the model parameters rather than determining a single "optimal" value for the model. In contrast to OLS, there is a posterior distribution for the model parameters that is proportional to the likelihood of the data multiplied by the prior probability of the parameters. One of the advantages of this approach is that if we have domain knowledge (Priors), or a an idea about the model parameters, we can include them in our model.The major  advantage of Bayesian processing is that you can incorporate the use of previous or assumed knowledge and update the current state of beliefs. You can incorporate prior information about a parameter and form a prior distribution for future analysis. One of the shortcomings of Bayesian analysis is that it does not tell you how to select a prior. There is no single correct way to choose a prior. This approach requires skills to translate subjective prior beliefs into a mathematically formulated prior. Any misunderstanding can generate misleading results.

- [R Example](https://www.r-bloggers.com/bayesian-linear-regression-analysis-without-tears-r/)

## Least-Angled Regression (LARS)

Least-Angled Regression (LARS), a new model selection algorithm, is a useful and less greedy version of traditional forward selection methods. This type of regression is useful when we have a high dimensional data. It's very similar to stepwise regression which finds out the best set of independent variables.

- [Python Example](https://plot.ly/scikit-learn/plot-lasso-lars/)

## Neural Network Regression

As the name suggests, neural networks are inspired by the brain. They form a network of interconnected nodes arranged in layers that make up a model. Neural networks are used to approzimate functions when the input data is too large for standard machine learning approaches.Fig. 8 represents the basic structure of a feed forward neural network. The input layer has number of nodes equal to a dimension of input data features. Each hidden layer consists of an arbitrary number of nodes. The number of the layers depends on the architecture and the scope of the problem.  And output layer consists of one node only if it is a regression problem. A neuron holds a number which represents the value of the corresponding feature of the input data, also known as activation. For each node of a single layer, input from each node of the previous layer is mixed in different proportions, and then passed into each node of the subsequent layer in a feed forward neural network, and so on until the signal reaches the final layer where a regression decision is made. All these are matrix operations.The questions then comes down to the network parameters which needs to be tuned such that it minimizes the loss between the predicted outcome and the true value. In large models, there can be millions of parameters to optimize. Gradient descent is used as the optimization function to adjust the weights/parameters in order to minimize the total error in the network. The gradient describes the relationship between the network’s error and a single weight, that is, how does the error vary as the weight is adjusted. As the training process continues, the network adjusts many weights/parameters such they can map the input data to produce an output which is as close as possible to the original output.Neural networks can run regression if given any prior information to predict a future event. For instance, you can predict heart attacks based on the vital stats data of a person. Moreover, you can also predict the likelihood that a customer will leave or not, based on web activity and metadata.

![Fig. 8: Neural Network Regression](https://datasciencedojo.com/wp-content/uploads/neural-network-regression.png)

- [Python Example](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)

- [Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/neural-network-regression)

## Locally Estimated Scatterplot Smoothing (LOESS)

LOESS is a highly flexible non-parametric regression technique. It makes as little assumptions as possible and tries to capture a general pattern. It is used to make an assessment of the relationship of two variables especially in large datasets.

- [R Example](http://r-statistics.co/Loess-Regression-With-R.html)

## Multivariate Adaptive Regression Splines (MARS)

MARS is a non-parametric model that fits a regression line in two phases. The first phase is a forward pass in which MARS starts with only an intercept and incrementally adds basis functions to it to improve the model. The brute force methodology of the first pass makes an overfit model which is pruned in the backward pass. In the backward pass any term from the model can be deleted.

- [R Example](http://uc-r.github.io/mars)

## Locally Weighted Regression (LWL)

This is a non-parametric model which makes local functions. It uses a set of weights, each for a subset of the data to make predictions on it. The use of higher weights for neighboring data points and lower weights for far away data, instead of using global patterns, makes it an accurate and flexible measure.

- [R Example](https://www.kdnuggets.com/2017/03/building-regression-models-support-vector-regression.html)

## Quantile Regression

Generally regression models predict the mean but this algorithm predicts the distribution of the data. It can be used to predict the distribution of prices given a certain input, for example what would be the 25th and 75th percentile of the distribution of a car price given some attributes.

- [Python Example](https://scikit-garden.github.io/examples/QuantileRegressionForests/)

## Principal Component Regression (PCR)

Principal Component Regression is an extension of Principal Component Analysis and Multiple Linear Regression. PCR models a target variable when there are a large number of predictor variables, and those predictors are highly correlated or even collinear. This method constructs new predictor variables, known as components, as linear combinations of the original predictor variables. PCR creates components to explain the observed variability in the predictor variables, without considering the target variable at all.In the first step, the principal components are calculated. The scores of the most important principal components are used as the basis for the multiple linear regression with the target variable. The most important point in PCR is the proper selection of the eigenvectors to be included. A plot of the eigenvalues usually indicates to the "best" number of eigenvectors.The benefit of PCR over multiple linear regression is that the noise remains in the residuals, since the eigenvectors with low eigenvalues represent only parts of the data with low variance. Moreover, the regression coefficients are more stable. This is because the eigenvectors are orthogonal to each other.

- [R Example](https://poissonisfish.wordpress.com/2017/01/23/principal-component-analysis-in-r/)

## Partial Least Squares Regression

Partial least squares regression (PLS regression) is developed from principal components regression. It works in a similar fashion as it finds a linear regression model by projecting the predicted variables and the predictor variables to a new space instead of finding hyperplanes of maximum variance between the target and predictor variables. While, PCR creates components to explain the observed variability in the predictors, without considering the target variable at all. PLS Regression, on the other hand, does take the response variable into account, and often leads to models that are able to fit the target variable with fewer components. However, it depends on the context of the model if using PLS Regression over PCR would offer a more parsimonious model.

- [R Example](https://rpubs.com/omicsdata/pls)

# neuralnetworks(11)

A neural network is an artificial model based on the human brain. These systems learn tasks by example without being told any specific rules. 

## Perceptron

A perceptron is a basic processing unit. The output of a perceptron is the the weighted sum of it's inputs and a bias unit, which acts as an intercept. A perceptron can define a decision boundary to separate two classes from each other. Multiple layers of perceptrons are combined to make much more powerful Artificial Neural Networks.

- [R Example](https://rpubs.com/FaiHas/197581)



## Multilayer Perceptron (MLP)

A single layer of perceptrons can only approximate a linear boundry in the data and cannot learn complex functions. A multilayer perceptron, besides input and output layers, also has hidden layers. This stack of layers allows it to learn non-linear decision boundaries in the data.MLP is considered a universal approximator because any arbitrary function can be learned from it using different assortments of layers and number of perceptrons in each layer. However, 'long and thin' networks are preferred over 'short and fat' networks.

- [Python Example](https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a)

## Recurrent Neural Network (RNN)

RNNs are used to learn sequences and temporal patterns. They achieve this by having self-connections, the perceptrons are connected to themselves, along with feed forward connections. These two types of connections allow them to learn both recurrency and a non-linear decision boundary.

![Fig.9: MLP and Partial Recurrency](https://datasciencedojo.com/wp-content/uploads/Introduction-to-machine-learning-recurrent-nn-by-Ethem-Alpaydin-2004.png)

- [R Example](https://cran.r-project.org/web/packages/rnn/vignettes/rnn.html)
- [Python Example](https://www.youtube.com/watch?v=BSpXCRTOLJA)

## Convolutional Neural Network (CNN)

Convolutional Neural Networks, AKA CNN or ConvNet, are very similar to traditional Neural Networks with minimal differences. The architechture for ConvNet assumes that images are to be encoded due to which the properties of the framework constructed are different from that of a plain vanilla neural network. Simple Neural Networks don’t scale well to full images because of their architectural design. A ConvNet is able to successfully capture the spatial and temporal information in an image just because of it's design and properties. Moreover, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, and depth. Each layer takes 3D data as input and transforms it to an output through a series of functions.These artificial neural networks are able to perform image classification, image recongnition, object detection and much more. These powerful algorithms can identify and label street signs, types of cancer, human faces, and many other aspects of life. CNN can also be applied in the field of text analytics to draw useful insights from the data.	

![Fig.10: Convolutional Neural Network Architecture](https://datasciencedojo.com/wp-content/uploads/convolutional-neural-network-architecture.png)

- [Python Example](https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html)

## Deep Belief Network (DBN)

DBN is a category of deep neural network which is comprised of multiple layers of graphical models having both directed and undirected edges. It is composed of multiple layers of hidden units, where each layer is connected with each other, but units are not. DBNs can learn the underlying structure of input and probabilistically reconstruct it. DBN does not use any labels. In other words, DBNs are generative models. In the training process of DBN, one layer is trained at a time. The first RBM is trained to re-construct its input as accurately as possible.The hidden layer of the first RBM is treated as the visible layer for the second layer and the second RBM is trained using the outputs from the first RBM. All these steps continue until all the layers of the DBN are trained. One thing to note about a DBN is that each RBM layer learns the entire input unlike convolutional nets, in which each layer detects unique patterns and later aggregates the result. A DBN fine-tunes the entire input in a sequence as the model is trained.Deep Belief Networks can be used in the field of Image Recognition, Video Sequence recognition, and Motion-capture data. 

![Fig. 11: Architecture of a Deep Belief Network](https://datasciencedojo.com/wp-content/uploads/DBN.png)

- [Python Example](https://medium.com/analytics-army/deep-belief-networks-an-introduction-1d52bb867a25)

## Hopfield Networks

Hopfield Networks are used to regain lost or distorted data. It is trained to memorize a pattern in data and reproduce it using a partial input. Each perceptron is an indivisible piece of information and will be connected to each other neuron. Thus all the neurons in it can be both input and output neurons.Hopfield networks are very computationally expensive as n inputs have n^2 weights. The network has to be trained till the weights stop changing.

- [Python Example](https://www.bonaccorso.eu/2017/09/20/ml-algorithms-addendum-hopfield-networks/)

## Learning Vector Quantization (LVQ)

LVQ addresses the drawback of KNN in that it needs to memorize the entire dataset for classification. LVQ uses a winner-takes-all strategy to identify representative vectors that are an approximation of the input space. The representatives are a form of low dimensionality compression.The model is prepared by using an input pattern to adjust the vectors most similar to it. Repeated performance of this procedure results in a distribution of vectors which provide a fair representation of the input space. Classification is performed by finding the Best Matching Unit (BMU) to the unlabeled input. The BMU has the least Euclidean distance to the input data, but other distance may also be used.The advantage of LVQ is that it is non-parametric - it does not make any assumptions about the data. However, the more complex the structure of the data, the more vectors and training iterations are required. It is recommended for robustness that the learning rate decays as training progresses and the number of passes for each learning rate is increased.

- [Python Tutorial](https://machinelearningmastery.com/implement-learning-vector-quantization-scratch-python/)

## Stacked Autoencoder

Stacked Autoeconders are mulitple layers of autoencoders that are trained in an unsupervised fashion individually. After this one final softmax layer is trained. These layers are combined after training to form a classifier which is trained a final time.

- [Python Example](http://deeplearning.net/tutorial/SdA.html)

## Boltzmann Machine

Boltzmann Machines are two layer neural networks which make stochastic decisions about the state of a system. A Boltzmann Machine does not discriminate between neurons, they are connected to each other. It was because of this they did not have much success.A Boltzmann Machine learns the distribution of data using the input and makes inferences on unseen data. It is a generative model - it does not expect input, it rather creates it.	

## Restricted Boltzmann Machine (RBM)

A Restricted Boltzmann Machine is called restricted because it has intra-layer communication. It can be used for feature selection, topic modelling and dimensionality reduction. In feed forward it learns the probability of neuron a being 1 given input x, and in back propagation it learns probability of x given a.It takes an input and tries to reconstruct it in forward and backward passes. Imagine a dataset of product purchases at a store. An RBM can be designed to take input of the products and connect them to nodes representing their categories. Thus, the RBM will learn a pattern between the category and purchase and make recommendations of the product.

- [Python Example](https://rubikscode.net/2018/10/22/implementing-restricted-boltzmann-machine-with-python-and-tensorflow/)

## Generative Adversarial Networks (GANs)

GANs are used for generating new data. A GAN comprises of 2 parts, a discriminator and a generator. The generator is like a reverse Convolutional Neural Network, it takes a small amount of data (random noise) and up scales it to generate input. The discriminator takes this input and predicts whether it belongs to the dataset.The two components are engaged in a zero-sum game to come up with new data, for example, GANs have been used to generate paintings.Python Tutorial

- [Python Tutorial](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f)

# anomalydetection (5)

Also known as outlier detection, anomaly detection is used to find rare occurrences or suspicious events in your data. The outliers typically point to a problem or rare event.

## Isolation Forest

Isolation Forests build a Random Forest in which each Decision Tree is grown randomly. At each node, it picks a feature randomly, then it picks a random threshold value (between the min and max value) to split the dataset in two. The dataset gradually gets chopped into pieces this way, until all instances end up isolated from the other instances.Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produces shorter path lengths for particular samples, they are highly likely to be anomalies.Python TutorialFig 12: IsolationForest Example

- [Python Tutorial](https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e)

  ![Fig 12: IsolationForest Example](https://scikit-learn.org/stable/_images/sphx_glr_plot_isolation_forest_001.png)



## Once Class SVM

One Class SVM is an anomaly detection technique which trains a Support Vector Machine on data with only one class. The SVM learns a boundary around the data and identifies which data points are far away from it. Data points which are abnormally far away from the data are considered outliers.

![Fig. 13: One-Class SVM](https://scikit-learn.org/stable/_images/sphx_glr_plot_oneclass_001.png)

- [R Tutorial](https://tsmatz.wordpress.com/2017/04/03/r-anomaly-detection-one-class-support-vector-machine-with-microsoftml-rxoneclasssvm/)
- [Python Tutorial](https://www.kaggle.com/amarnayak/once-class-svm-to-detect-anomaly)

## PCA-Based Anomaly Detection

PCA-Based Anomaly Detection uses distance metrics to differentiate between normal and anomalous behavior. Data points which are far apart from most of the data are classified as anomalous.

- [Python Example](https://www.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/ch04.html)

## Fast-MCD

The Fast-MCD (minimum covariance determinant) algorithm is used for outlier detection. It assumes that the normal instances, also called inliers, are generated from a single Gaussian distribution and not a mixture, but it also assumes that the dataset is contaminated with outliers that were not generated from this Gaussian distribution. When it estimates the parameters of the Gaussian distribution, it is careful to ignore the instances that are most likely outliers. This makes it better at identifying the outliers. 

- [R Example](https://www.rdocumentation.org/packages/robustbase/versions/0.93-5/topics/covMcd)

## Local Outlier Factor (LOF)

LOF is an unsupervised algorithm and is used for anomaly/outlier detection which is based on the local density estimation. It compares the density of instances around a given instance to the density around its neighbors. An anomaly is often more isolated than its k nearest neighbors. If the density of a point is much smaller than the densities of its neighbors (LOF ≫1), the point is far from dense areas and, hence, an outlier. In short, we can say that the density around an outlier object is significantly different from the density around its neighbors.It can be summarized in three steps:

- [Python Tutorial](https://medium.com/@mtngt/local-outlier-factor-simple-python-example-8925dad97fe6)

# dimensionalityreduction(17)

With some problems, especially classification, there can be so many variables, or features, that it is difficult to visualize your data. The correlation amongst your features creates redundancies, and that's where dimensionality reduction comes in. Dimensionality Reduction reduces the number of random variables you're working with. 

## Singular Value Decomposition (SVD)

This is a form of matrix analysis that leads to a low-dimensional representation of a high-dimensional matrix. SVD allows an approximate representation of any matrix, and also makes it easy to eliminate the less important parts of that representation to produce an approximate representation with any desired number of dimensions.Suppose we want to represent a very large and complex matrix using some smaller matrix representation then SVD can factorize an m x n matrix, M, of real or complex values into three component matrices, where the factorization has the form USV. The best way to reduce the dimensionality of the three matrices is to set the smallest of the singular values to zero. If we set a particular number of smallest singular values to 0, then we can also eliminate the corresponding columns. The choice of the lowest singular values to drop when we reduce the number of dimensions can be shown to minimize the root-mean-square error between the original matrix M and its approximation. A useful rule of thumb is to retain enough singular values to make up 90% of the energy. That is, the sum of the squares of the retained singular values should be at least 90% of the sum of the squares of all the singular values. It is also possible to reconstruct the approximation of the original matrix M using U, S , and V.SVD is used in the field of predictive analytics. Normally, we would want to remove a number of columns from the data since a greater number of columns increases the time taken to build a model. Eliminating the least important data gives us a smaller representation that closely approximates the original matrix. If some columns are redundant in the information they provide then this means those columns contribute noise to the model and reduce predictive accuracy. Dimensionality reduction can be achieved by simply dropping these extra columns. The resulting transformed data set can be provided to machine learning algorithms to yield much faster and accurate models.

![Fig. 14: SVD](https://datasciencedojo.com/wp-content/uploads/singular-value-decomposition-svd.png)

- [R Example](https://www.displayr.com/singular-value-decomposition-in-r/)

## Forward Feature Selection

Forward Selection is performed by starting with 1 or a few features initially and creating a model. Another feature is repeatedly added to improve the model till the required level of accuracy is achieved. This is a rather slow approach and impractical when there are a large number of features available.

- [Python Example](https://www.kdnuggets.com/2018/06/step-forward-feature-selection-python.html)

## Backward Feature Elemination

Backward Elimination is performed by starting with all or most of the features to be used for the model and eliminating the features one at a time to improve the model. The removed features are indiscriminant and add confusion to the model. Statistical techniques such as R squared metric and statistical tests can be used to decide which features to remove.

- [Python Example](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)

## Subset Selection

In this technique a subset of features is selected by manual trial. Variables are added and removed such that the Error term is reduced. An exhaustive approach would take 2^n models, where n is the number of features – therefore a heuristic technique is used because a thorough approach is too expensive.There are three methodologies – forward selection, backward selection and floating search. Forward selection is performed by incrementally adding a variable to the model to reduce the error. Backward selection is performed by starting with all the variables and reducing them stepwise to improve the model. Floating Search uses a back and forth approach to add and reduce variables to form different combinations.

- [R Example](http://www.science.smith.edu/~jcrouser/SDS293/labs/lab8-r.html)

## Principal Component Analysis (PCA)

PCA is a projection technique which find a projection of the data in a smaller dimension. The idea is to find an axis in the data with highest variance and to map the data along that axis.In figure 15, the data along vector 1 shows a higher variance than vector 2. Therefore, vector 1 will be preferred and chosen as the first principle component. The axis has been rotated in the direction of highest variance. We have thus reduced the dimensionality from two (X1 and X2) to one (PC 1).PCA is useful in cases where the dimensions are highly correlated. For example, pixels in images have a high correlation with each other, here will will prove a significant gain my reducing the dimension. However, if the features are not correlated to each other than the dimension will be the almost the same in quantity after PCA.Fig. 15: Original vs Principal Component R Tutorial

![Fig. 15: Original vs Principal Component ](http://datasciencedojo.com/wp-content/uploads/principle-component-analysis-pca.png)

- [R Tutorial](https://www.r-bloggers.com/principal-component-analysis-in-r/)

## Partial Least Squares Regression (PLSR)

Partial least squares regression (PLS regression) is developed from principal components regression. It works in a similar fashion as it finds a linear regression model by projecting the predicted variables and the predictor variables to a new space instead of finding hyperplanes of maximum variance between the target and predictor variables. While, PCR creates components to explain the observed variability in the predictor variables, without considering the target variable at all, PLS Regression, on the other hand,  does take the response variable into account, and therefore often leads to models that are able to fit the target variable with fewer components. However, it depends on the context of the model if using PLS Regression over PCR would offer a more parsimonious model.

- [R Example](https://rpubs.com/omicsdata/pls)

## Latent Dirichlet Analysis (LDA)

Latent Dirichlet Allocation (LDA) is one of the most popular techniques used for topic modelling. Topic modelling is a process to automatically identify topics present in a text object.A latent Dirichlet allocation model discovers underlying topics in a collection of documents and infers word probabilities in topics. LDA treats documents as probabilistic distribution sets of words or topics. These topics are not strongly defined – as they are identified based on the likelihood of co-occurrences of words contained in them.The basic idea is that documents are represented as random mixtures over latent topics, where each topic is characterized by a distribution over words. Given a dataset of documents, LDA backtracks and tries to figure out what topics would create those documents in the first place. The goal of LDA is to map all the documents to the topics in a way, such that the words in each document are mostly captured by those imaginary topics.A collection of documents is represented as a document-term matrix. LDA converts this document-term matrix into 2 lower dimensional matrices, where one is a document-topics matrix and the other is a topic-terms matrix. LDA then makes use of sampling techniques in order to improve these matrices. A steady state is achieved where the document topic and topic term distributions are fairly good. As a result, it builds a topic per document model and words per topic model, modeled as Dirichlet distributions.

- [R Example](https://www.tidytextmining.com/topicmodeling.html)

## Regularized Discriminant Analysis (RDA)

The regularized discriminant analysis (RDA) is a generalization of the linear discriminant analysis (LDA) and the quadratic discriminant analysis (QDA). RDA differs from discriminant analysis in a manner that it estimates the covariance in a new way, which combines the covariance of QDA with the covariance of LDA using a tuning parameter. Since RDA is a regularization technique, it is particularly useful when there are many features that are potentially correlated.

- [R Example](https://daviddalpiaz.github.io/r4sl/regularized-discriminant-analysis.html)

## t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a non-linear dimensionality reduction algorithm used for exploring high-dimensional data. It maps multi-dimensional data to lower dimensions which are easy to visualize.This algorithm calculates probability of similarity of points in high-dimensional space and in the low dimensional space. It then tries to optimize these two similarity measures using a cost function. To measure the minimization of the sum of difference of conditional probability, t-SNE minimizes the sum of Kullback-Leibler divergence of data points using a gradient descent method. t-SNE minimizes the divergence between two distributions: a distribution that measures pairwise similarities of the high-dimensional points and a distribution that measures pairwise similarities of the corresponding low-dimensional points. Using this technique, t-SNE can find patterns in the data by identifying clusters based on similarity of data points with multiple features.t-SNE stands out from all the other dimensionality reduction techniques since it is not limited to linear projections so it is suitable for all sorts of datasets. 

- [R and Python Examples](https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/)

## Factor Analysis

Factor Analysis is designed on the premise that there are latent factors which give origin to the available data that are not observed. In PCA, we create new variables with the available ones, here we treat the data as created variables and try to reach the original ones – thus reversing the direction of PCA.If there is a group of variables that are highly correlated, there is an underlying factor that causes that and can be used as a representative variable. Similarly, the other variables can also be grouped and these groups can be represented using such representative variables.Factor analysis can also be used for knowledge extraction, to find the relevant and discriminant piece of information.

- [R Example](https://www.promptcloud.com/blog/exploratory-factor-analysis-in-r/)

## Multidimensional Scaling (MDS)

Multidimensional Scaling (MDS) computes the pairwise distances between data points in the original dimensions of the data. The data points are mapped on the a lower dimension space, like the Euclidean Space, such that the paints with low pairwise distances in higher dimension are also close in the lower dimension and points which are far apart in higher dimension, are also apart in lower dimension.The pitfall of this algorithm can be seen in the analogy of geography. Locations which are far apart in road distance due to mountains or rough terrains, but close by in bird-flight path will be mapped far apart by MDS because of the high value of the pairwise distance.

![Fig 15: Map of Europe drawn by MDS](https://datasciencedojo.com/wp-content/uploads/Diagram-from-the-introduction-to-machine-learning-by-Ethem-Alpaydin.png)

- [R Example](http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/122-multidimensional-scaling-essentials-algorithms-and-r-code/)

## AutoEncoder

A tool for dimensionality reduction, an autoencoder has as many outputs as inputs and it is forced to find the best representation of the inputs in the hidden layer. There are fewer perceptrons in the hidden layer, which implies dimensionality reduction. Once training is complete, the first layer from the input layer to the hidden layer acts as an encoder which finds a lower dimension representation of the data. The decoder is from the layer after the hidden layer to the output layer.The encoder can be used to pass data and find a lower dimension representation for dimension reduction.

- [Python Example](https://blog.keras.io/building-autoencoders-in-keras.html)

## Independent Component Analysis (ICA)

ICA solves the cocktail party problem. At a cocktail party, one is able to seperate the voice of any one person from the voices in the background. Computers are not as efficient at separating the noise from signal as the human brain, but ICA can solve this problem if the data is not Gaussian.ICA assumes independence among the variables in the data. It also assumes that the mixing of the noise and signal is linear, and the source singal has a non-gaussian distribution.

- [R Example](https://rpubs.com/skydome20/93614)

## Isomap

Isomap (Isometric Mapping) computes the geodesic distances between data points and maps those distances in a Euclidean space to create a lower dimension mapping of the same data.Isomap offers the advantage of using global patterns by first making a neighborhood graph using euclidean distances and then computes graph distances between the nodes. Thus, it uses local information to find global mappings.

- [Python Example](http://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/)

## Local Linear Embedding (LLE)

LLE reduces the dimension of the data such that neighbourhood information (topology) is intact. Points that are far apart in high dimension should also be far apart in lower dimension. LLE assumes that data is on a smooth surface without abrupt holes and that it is well sampled (dense).LLE works by creating a neighbourhood graph of the dataset and computing a local weight matrix using which it regenerates the data in lower dimension. This local weight matrix allows it to maintain the topology of the data.

- [R Example](http://rstudio-pubs-static.s3.amazonaws.com/94107_913ae6a497fc408a91a2529b6c57f791.html)

## Locality-Sensitive Hashing

This technique uses a hash function to determine the similarity of the data. A hash function provide a lower dimensional unique value for an input and used for indexing in databases. Two similar values will give a similar hash value which is used by this technique to determine which data points are neighbours an which are far apart to produce a lower dimensional version of the input data set.

- [R Example](https://cran.r-project.org/web/packages/textreuse/vignettes/textreuse-minhash.html)

## Sammon Mapping

Sammon Mapping creates a projection of the data such that geometric relations between data points are maintained to the highest extent. It creates a new dataset using the pairwise distances between points. Sammon mapping is frequently used in image recognition tasks.

![Fig. 16: Sammon Mapping vs. PCA Projection](https://datasciencedojo.com/wp-content/uploads/sammon-mapping-vs-pca-projection.png)

- [Paul Henderson](http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0910/henderson.pdf)
- [Python Example](https://datawarrior.wordpress.com/2016/10/23/sammon-embedding/)

# ensemble(11)

Ensemble learning methods are meta-algorithms that combine several machine learning methods into a single predictive model to increase the overall performance. 

## Random Forest

A random forest is comprised of a set of decision trees, each of which is trained on a random subset of the training data. These trees predictions can then be aggregated to provide a single prediction from a series of predictions.To build a random forest, you need to choose the total number of trees and the number of samples for each individual tree. Later, for each tree, the set number of samples with replacement and features are selected to train the decision tree using this data.The outputs from all the seperate models are aggregated into a single prediction as part of the final model. In terms of regression, the output is simply the average of predicted outcome values. In terms of classification, the category with the highest frequency output is chosen.The bootstrapping and feature bagging process outputs varieties of different decision trees rather than just a single tree applied to all of the data.Using this approach, the models that were trained without some features will be able to make predictions in aggregated models even with missing data. Moreover, each model trained with different subsets of data will be able to make decisions based on different structure of the underlysing data/population. Hence, in aggregated model they will be able to make prediction even when the training data doesn’t look exactly like what we’re trying to predict. 

- [Python Example](https://medium.com/datadriveninvestor/random-forest-regression-9871bc9a25eb)

## Bagging (Bootstrap Aggregation)

Bagging (Bootstrap Aggregation) is used when we want to reduce the variance (over fitting) of a decision tree. Bagging comprises of the following steps:Bootstrap SamplingSeveral subsets of data can be obtained from the training data chosen randomly with replacement. This collection of data will be used to train decision trees. Bagging will construct n decision trees using bootstrap sampling of the training data. As a result, we will have an ensemble of different models at the end.AggregationThe outputs from all the seperate models are aggregated into a single prediction as part of the final model. In terms of regression, the output is simply the average of predicted outcome values. In terms of classification, the category with the highest frequency output is chosen. Unlike boosting, bagging involves the training a bunch of individual models in a parallel way. The advantage of using Bootstrap aggregation is that it allows the variance of the model to be reduced by averaging multiple estimates that are measured from random samples of a population data.

- [R Example](http://rpubs.com/kangrinboqe/268745)

## AdaBoost

AdaBoost is an iterative ensemble method. It builds a strong classifier by combining multiple weak performing classifiers.The final classifier is the weighted combination of several weak classifiers. It fits a sequence of weak learners on different weighted training data. If prediction is incorrect using the first learner, then it gives higher weight to observation which have been predicted incorrectly. Being an iterative process, it continues to add learner(s) until a limit is reached in the number of models or accuracy. You can see this process represented in the AdaBoost Figure.Initially, AdaBoost selects a training subset randomly and gives equal weight to each observation. If prediction is incorrect using the first learner then it gives higher weight to observation which have been predicted incorrectly. The model is iteratively training by selecting the training set based on the accurate prediction of the last training. Being an iterative process, the model continues to add multiple learners until a limit is reached in the number of models or accuracy.It is possible to use any base classifier with AdaBoost. This algorithm is not prone to overfitting. AdaBoost is easy to implement. One of the downsides of AdaBoost is that it is highly affected by outliers because it tries to fit each point perfectly. It is computationally slower as compared to XGBoost. You can use it both for classification and regression problem. 

![Fig. 17: AdaBoost](https://datasciencedojo.com/wp-content/uploads/adaboost.png)

- [R Tutorial](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)

## Gradient Boosting

Gradient boosting is a method in which we re-imagine the boosting problem as an optimisation problem, where we take up a loss function and try to optimise it.Gradient boosting involves 3 core elements: a weak learner to make predictions, a loss function to be optimized, and an additive model to add to the weak learners to minimize the loss function.This algorithm trains various models sequentially. Decision trees are used as the base weak learner in gradient boosting. Trees are added one at a time, and existing trees in the model are not changed. Each new tree helps to correct errors made by previously trained tree. A gradient descent procedure is used to minimize the loss when adding trees. After calculating error or loss, the parameters of the tree are modified to minimize that error. Gradient Boosting often provides predictive accuracy that cannot be surpassed. These machines can optimize different loss functions depending on the problem type which makes it felxible. There is no data pre-processing required as it also handles missing data.One of the applications of Gradient Boosting Machine is anomaly detection in supervised learning settings where data is often highly unbalanced such as DNA sequences, credit card transactions or cyber security. One of the drawbacks of GBMs is that they are more sensitive to overfitting if the data is noisy and are also computationally expensive which can be time and memory exhaustive.

- [R Tutorial](https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab)

## Gradient Boosted Regression Trees

Gradient Boosted Regression Trees (GBRT) are a flexible, non-parametric learning technique for classification and regression, and are one of the most effective machine learning models for predictive analytics. Boosted regression trees combine the strengths of two algorithms which include regression trees and boosting methods. Boosted regression trees incorporate important advantages of tree-based methods, handling different types of predictor variables and accommodating missing data. They have no need for prior data transformation or elimination of outliers, can fit complex nonlinear relationships, and automatically handle interaction effects between predictors. 

- [Python Example](https://scikit-learn.org/stable/modules/ensemble.html)

## XGBoost (Extreme Gradient Boosting

"XGBoost is similar to gradient boosting framework but it improves upon the base GBM architechture by using system optimization and algorithmic improvements.System optimizations:
Parallelization: It executes the sequential tree building using parallelized implementation. 
Hardware: It uses the hardware resources efficiently by allocating internal buffers in each thread to store gradient statistics.Tree Pruning: XGBoost uses ‘max_depth’ parameter instead of criterion first, and starts pruning trees backward. This ‘depth-first’ approach improves computational performance significantly.Algorithmic Improvements:
Regularization: It penalizes more complex models through both LASSO (L1) and Ridge (L2) regularization to prevent overfitting.Sparsity Awareness: Handles different types of sparsity patterns in the data more efficiently.Cross-validation: The algorithm comes with built-in cross-validation method at each iteration, taking away the need to explicitly program this search and to specify the exact number of boosting iterations required in a single run.Due to it's computational complexity and ease of implementation, XGBoost is used widely over Gradient Boosting."

- [R Tutorial](https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/)

## Voting Classifier

A voting classifier combines the results of several classifiers to predict the class labels. It is one of the simplest ensemble methods. The voting classifier usually achieves better results than the best classifier in the ensemble. A hard-voting classifier uses the majority vote to predict the class labels. Whereas, a soft-voting classifier will use the average predicted probabilities to predict the labels, however, this can only be possible if all individual classifiers can predict class probabilities.The voting classifier can balance out the individual weakness of each classifier used. It will be beneficial to include diverse classifiers so that models which fall prey to similar types of errors do not aggregate the errors. As an example, one can train a logistic regression, a random forest classifier a naïve bayes classifier and a support vector classifier. To predict the label, the class that receives the highest number of votes from all of the 4 classifiers will be the predicted class of the ensemble (Voting classifier).

- [Python Tutorial](http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/)

## Extremely Randomized Trees

Extremely Randomized Trees (also known as Extra-Trees) increases the randomness of Random Forest algorithms and moves a step further. As in random forests, a random subset of candidate features is used, but instead of looking for the most discriminating thresholds, thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule.This trades more bias for a lower variance. It also makes Extra-Trees much faster to train than regular Random Forests since finding the best possible threshold for each feature at every node is one of the most time-consuming tasks of growing a tree. One can use it for both regression and classification.

- [R Example](https://daviddalpiaz.github.io/stat432sp18/lab/enslab/enslab.html)

## Boosted Decision Tree

Boosted Decision Trees are a collection of weak decision trees which are used in congregation to make a strong learner. The other decision trees are called weak because they have lesser ability than the full model and use a simpler model. Each weak decision tree is trained to address the error of the previous tree to finally come up with a robust model.

- [R Example](https://www.r-bloggers.com/gradient-boosting-in-r/)

## Category Boosting (CatBoost)

CatBoost is a fast, scalable, high performance algorithm for gradient boosting on decision trees. It can work with diverse data types to help solve a wide range of problems that businesses face today. Catboost achieves the best results on the benchmark.Catboost is built with a similar approach and attributes as with Gradient Boost Decision Tree models. The feature that separates CatBoost algorithm from rest is its unbiased boosting with categorical variables. Its power lies in its categorical features preprocessing, prediction time and model analysis.Catboost introduces two critical algorithmic advances - the implementation of ordered boosting, a permutation-driven alternative to the classic algorithm, and an innovative algorithm for processing categorical features.CatBoost handles data very efficiently, few tweaks can be made to increase efficiency like choosing the mode according to data. However, Catboost’s training and optimization times is considerably high.

- [R Tutorial](https://www.kaggle.com/slickwilly/simple-catboost-in-r)

## Stacked Generalization (Stacking)

Stacking is an ensemble method where a new model is trained to combine the predictions from two or more models already trained on a dataset. It is based on a simple idea: instead of using trivial ensemble functions to aggregate the predictions of all predictors in an ensemble, stacking would train a model to perform this aggregation. The idea is that you can attack a learning problem with different types of models which are capable to learn some part of the problem, but not the whole space of the problem.The procedure starts with splitting the training set into two disjoint sets. Following this we would train several base learners on the first part and test the base learners on the second part. Using these predictions as the inputs, and the correct responses as the outputs, we’ll train a higher-level learner.For example, for a classification problem, we can choose as weak learners a KNN classifier, a logistic regression and an SVM, and decide to learn a neural network as meta-model. Then, the neural network will take as inputs the outputs of our three weak learners and will learn to return final predictions based on it.It is typical to use a simple linear method to combine the predictions for sub models such as simple averaging or voting, to a weighted sum using linear regression or logistic regression. It is important that sub-models produce different predictions, so-called uncorrelated predictions. Stacking is one of the most efficient techniques used in winning data science competitions.

- [Python Tutorial](https://machinelearningmastery.com/implementing-stacked-scratch-python)

# clustering(14)

In supervised learning, we know the labels of the data points and their distribution. However, the labels may not always be known. Clustering is the practice of assigning labels to unlabeled data using the patterns that exist in it. Clustering can either be semi-parametric or probabilistic. 

## K-Means Clustering

K-Means Clustering is an iterative algorithm which starts of with k random numbers used as mean values to define clusters. Data points belong to the cluster defined by the mean value to which they are closest. This mean value co-ordinate is called the centroid.Iteratively, the mean value of the data points of each cluster is computed and the new mean values are used to restart the process till mean stop changing. The disadvantage of K-Means is that it a local search procedure and could miss global patterns.The k initial centroids can be randomly selected. Another approach of determining k is to compute the mean of the entire dataset and add k random co-ordinates to it to make k initial points. Another approach is to determine the principle component of the data and divide into k equal partitions. The mean of each partition can be used as initial centroids.

- [Python Example](https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/)

## K-Medians Clustering

K-Medians uses absolute deviations (Manhattan Distance) to form k clusters in the data. The centroid of the clusters is the median of the data points in the cluster. This technique is the same as K-Means but more robust towards outliers because of the use of median not mean, because K-Means optimizes the squared distances.Consider a list of numbers: 3, 3, 3, 9. It's median is 3 and mean is 4.5. Thus, we see that use of median prevents the effect of outliers.

- [Python Example](https://gist.github.com/mblondel/1451300)

## Mean Shift Clustering

Mean Shift is a hierarchical clustering algorithm. It is a sliding-window-based algorithm that attempts to find dense areas of data points. Mean shift considers the feature space as sampled from the underlying probability density function. For each data point, Mean shift associates it with the nearby peak of the dataset's probability density function. Given a set of data points, the algorithm iteratively assigns each data point towards the closest cluster centroid. A window size is determined and a mean of the data points within the window is calculated. The direction to the closest cluster centroid is determined by where most of the points nearby are at. So after each iteration, each data point will move closer to where the most points are at, which leads to the cluster center.Then, the window is shifted to the newly calculated mean and this process is repeated until convergence. When the algorithm stops, each point is assigned to a cluster.Mean shift can be used as an  image segmentation algorithm. The idea is that similar colors are grouped to use the same color. This can be accomplished by clustering the pixels in the image. This algorithm is really simple since there is only one parameter to control which is the sliding window size. You don't need to know the number of categories (clusters) before applying this algorithm, as opposed to K-Means. The downside to Mean Shift is it's computationally expensive — O(n²). The selection of the window size can be non-trivial. Also, it does not scale well with dimension of feature space. 

- [Python Example](https://pythonprogramming.net/mean-shift-from-scratch-python-machine-learning-tutorial/)

## K-Modes Clustering

A lot of data in real world data is categorical, such as gender and profession, and, unlike numeric data, categorical data is discrete and unordered. Therefore, the clustering algorithms for numeric data cannot be used for categorical data. K-Means cannot handle categorical data since mapping the categorical values to 1/0 cannot generate quality clusters for high dimensional data so instead we can land onto K-Modes.The K-Modes approach modifies the standard K-Means process for clustering categorical data by replacing the Euclidean distance function with the simple matching dissimilarity measure, using modes to represent cluster centers and updating modes with the most frequent categorical values in each of iterations of the clustering process. These modifications guarantee that the clustering process converges to a local minimal result. The number of modes will be equal to the number of clusters  required, since they act as centroids. The dissimilarity metric used for K-Modes is the Hamming distance from information theory which can be seen in Fig. 25. Here, x and y are the values of attribute j in object X and Y. The larger the number of mismatches of categorical values between X and Y is, the more dissimilar the two objects. In case of categorical dataset, the mode of an attribute is either “1” or “0,” whichever is more common in the cluster. The mode vector of a cluster minimizes the sum of the distances between each object in the cluster and the cluster centerThe K-Modes clustering process consists of the following steps:

![Fig. 18: Hamming Distance](https://datasciencedojo.com/wp-content/uploads/k-mode-hammer-distance.png)

[Python Example](https://pypi.org/project/kmodes/)

## Fuzzy K-Modes

The Fuzzy K-Modes clustering algorithm is an extension to K-Modes. Instead of assigning each object to one cluster, the Fuzzy K-Modes clustering algorithm calculates a cluster membership degree value for each object to each cluster. Similar to the Fuzzy K-Means, this is achieved by introducing the fuzziness factor in the objective function.The Fuzzy K-Modes clustering algorithm has found new applications in bioinformatics. It can improve the clustering result whenever the inherent clusters overlap in a data set.

- [Python Example](https://github.com/medhini/Genetic-Algorithm-Fuzzy-K-Modes)

## Fuzzy C-Means

Fuzzy C-Means is a probabilistic version of K-Means clustering. It associates all data points to all clusters such that the sum of all the associations is 1. The impact is that all clusters have a continuous (as opposed to discrete as in K-Means) association to each cluster relative to each other cluster.The algorithm iteratively assigns and computes the centroids of the clusters the same as K-Means till either criterion function is optimized of the convergence falls below a predetermined threshold value.The advantages of this algorithm are that it is not stringent like K-Means in assigning and works well for over lapping datasets. However it has the same disadvantage as K-Means of having a prior assumption of the number of clusters. Also, a low threshold value gives better results but is more computationally costly.

- [Python Example](https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html)

## Mini Batch K-Means Clustering

Mini Batch K-Means uses a random subset of the entire data set to perform the K-Means algorithm. The provides the benefit of saving computational power and memory requirements are reduced, thus saving hardware costs or time (or a combination of both).There is, however, a loss in overall quality, but an extensive study as shows that the loss in quality is not substantial.

![Fig. 19: Difference between K-Means and Mini-Batch Graph](https://media.geeksforgeeks.org/wp-content/uploads/20190510070216/8fe10fb0-438d-4706-8fb0-f3ae95f35652.png)

- [Python Example](https://www.geeksforgeeks.org/ml-mini-batch-k-means-clustering-algorithm/)

## Hierarchical Clustering

Hierarchical Clustering uses the approach of finding groups in the data such that the instances are more similar to each other than to instances in other groups. This measure of similarity is generally a Euclidean distance between the data points, but Citi-block and Geodesic distances can also be used.The data is broken down into clusters in a hierarchical fashion. The number of clusters is 0 at the top and maximum at the bottom. The optimum number of clusters is selected from this hierarchy.

- [R Example](https://uc-r.github.io/hc_clustering)

## Expectation Maximization

Expectation Maximization uses a Maximum Likelihood Estimate system and is a three step procedure. The first step is Estimation - to conjecture parameters and a probability distribution for the data. The next step is to feed data into the model. The 3rd step is Maximization - to tweak the parameters of the model to include the new data. These three steps are repeated iteratively to improve the model.

- [R Example](http://rstudio-pubs-static.s3.amazonaws.com/154174_78c021bc71ab42f8add0b2966938a3b8.html)

## DBSCAN

DBSCAN stands for Density-based spatial clustering of applications with noise. Points that are a x distance from each other are a dense region and form a set of core points. Points that are x distance from each other, both core and non-core, form a cluster. Points that are not reachable from any core points are noise points.Density-Based Spatial Clustering of Applications with Noise is a density based clustering algorithm which identifies dense regions in the data as clusters. Dense regions are defined as areas in which points are reachable by each other. The algorithm uses two parameters, epsilon, and minimum points.Two data points are within reach of each other if their distance is less than epsilon. A cluster also needs to have a minimum number of points to be considered a cluster. Points which have the minimum number of points within epsilon distance are called core points.Points that are not reachable by any cluster are Noise points.DBSCAN's density based design makes it robust to outliers. However, it does not work well when working with clusters of varying density.

- [Python Example](https://medium.com/@elutins/dbscan-what-is-it-when-to-use-it-how-to-use-it-8bd506293818)

## Minimum Spanning Trees

The minimum spanning tree clustering algorithm is capable of detecting clusters with irregular boundaries. The MST based clustering method can identify clusters of arbitrary shape by removing inconsistent edges. The clustering algorithm constructs MST using Kruskal algorithm and then sets a threshold value and step size. It then removes those edges  from the MST, whose lengths are greater than the threshold value. A ratio between the intra-cluster distance and inter-cluster distance is calculated. Then, the threshold value is updated by incrementing the step size. At each new threshold value, the steps are repeated. The algorithm stops when no more edges can be removed from the tree. At this point, the minimum  value of the ratio can be checked and the clusters can be formed corresponding to the threshold value.MST searches for that optimum value of the threshold for which the Intra and Inter distance ratio is minimum. Generally, MST comparatively performs better than the k-Means algorithm for clustering.

- [Python Tutorial](https://slicematrix.github.io/mst_stock_market.html)

## Quality Threshold

Quality Threshold uses a minimum distance a point has to be away from a cluster to be a member and a minimum number of points for each cluster. Points are assigned clusters till the point and the cluster qualify these two criteria. Thus the first cluster is made and the process is repeated on the points which were not within distance and beyond the minimum number to form another cluster.The advantage of this algorithm is that quality of clusters is guaranteed and unlike K-Means the number of clusters does not have to be fixed apriori. The approach is also exhaustive and candidate clusters for all data points are considered.The exhaustive approach has the disadvantage of being computationally intense and time consuming. There is also the requirement of selecting the distance and minimum number apriori.

- [Python Example](https://github.com/melvrl13/python-quality-threshold/blob/master/QT.py)

## Gaussian Mixture Model (GMM)

A Gaussian mixture model (GMM) is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown. In this approach we describe each cluster by its centroid (mean), covariance , and the size of the cluster(Weight). All the instances generated from a single Gaussian distribution form a cluster where each cluster can have a different shape, size, density and orientation.GMMs have been used for feature extraction from speech data and have also been used extensively in object tracking of multiple objects. The parameters for Gaussian mixture models are derived either from maximum a posteriori estimation or an iterative expectation-maximization algorithm from a prior model which is well trained.

- [R Tutorial](http://tinyheero.github.io/2015/10/13/mixture-model.html)

## Spectral Clustering

Spectral clustering has become a promising alternative to traditional clustering algorithms due to its simple implementation and promising performance in many graph-based clustering. The goal of spectral clustering is to cluster data that is connected but not necessarily compact or clustered within convex boundaries. This algorithm relies on the power of graphs and the proximity between the data points in order to cluster them. This makes it possible to avoid the sphere shape cluster that the K-Means algorithm forces us to assume. As a result, spectral clustering usually outperforms K-Means algorithm.In practice Spectral Clustering is very useful when the structure of the individual clusters is highly non-convex or more generally when a measure of the center and spread of the cluster is not a suitable description of the complete cluster. For instance, when clusters are nested circles on the 2D plane.Spectral Clustering requires the number of clusters to be specified. It works well for a small number of clusters but is not advised when using many clusters.

- [Python Tutorial](https://medium.com/@tomernahshon/spectral-clustering-from-scratch-38c68968eae0)

# associationruleanalysis(2)

Association rule analysis is a technique to uncover how items are associated with each other. 

## Apriori

This algorithm is one of the basic approaches used in mining frequent patterns from datasets. It is one of the fundamental algorithms used in the market-basket analysis. You can use this to find the groups of items that occur together frequently in a shopping dataset which can help buisnesses find ways to promote their products. Apriori works by finding the counts of all the items in the dataset and filtering out items that do not occur frequently. The counts can be extended to a pair of items and later on to the count of itemsets till size N and then filtering out the infrequent itemsets. One thing to note here is that the frequent pairs are those where both items in the pair are frequent items.The advantage of this algorithm is that it saves a lot of time by cutting down on the number of itemsets that it builds and counts.

- [R Example](http://r-statistics.co/Association-Mining-With-R.html)

## Eclat

The ECLAT algorithm stands for Equivalence Class Clustering and bottom-up Lattice Traversal. While the Apriori algorithm works in a horizontal sense imitating the Breadth-First Search of a graph, the ECLAT algorithm works in a vertical manner just like the Depth-First Search of a graph. Due to this vertical approach, ECLAT is faster and scalable than the Apriori algorithm.ECLAT is superior over Apriori because of memory (Since the ECLAT algorithm uses a Depth-First Search approach, it uses less memory than Apriori algorithm) and computations (The ECLAT algorithm does not involve the repeated scanning of the data to compute the support values).

- [R Example](http://r-statistics.co/Association-Mining-With-R.html)

# regularization(3)

Regularization is used to prevent overfitting. Overfitting means a machine learning algorithm has fit the data set too strongly such that it has high accuracy in it but does not perform well on unseen data. 

## LASSO Regularization (Least Absolute Shrinkage and Selection Operator)

LASSO regularization adds the sum of the absolute values of the coefficients of the model to the cost function. This also acts as a form of feature selection as the coefficients may become 0 and only the coefficients of the variables which are discriminative stay. LASSO works well if few features affect the predictor variable (with a high coefficient) and others are close to zero.

- [Lasso and Ridge in Python](https://www.kaggle.com/jmataya/regularization-with-lasso-and-ridge)

## Ridge Regularization

Ridge regularization works by adding the square of the coefficients of the model to the cost function. The coefficients of correlated features becomes similar. Ridge works well if there are many large coefficients of the same value, that is, many features have a strong impact on the predicted variable.

- [Lasso and Ridge in Python](https://www.kaggle.com/jmataya/regularization-with-lasso-and-ridge)

## Elastic Net Regularization

Elastic Net is a combination of the penalties of LASSO and Ridge. It picks out a group of independent variables which are correlated and if there is a strong predicting power in them all of them will be used.

- [R Tutorial](https://www.r-bloggers.com/variable-selection-with-elastic-net/)