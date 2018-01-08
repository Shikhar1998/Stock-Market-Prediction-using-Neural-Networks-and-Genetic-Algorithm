# Stock Market Prediction using Neural Networks and Genetic Algorithm
This module employs Neural Networks and Genetic Algorithm to predict the future values of stock market.
The test data used for simulation is from the Bombay Stock Exchange(BSE) for the past 40 years. 

## 1. Introduction

Stock market prediction is the “act of determining” the future value of a company stock or other financial instrument traded on an exchange. The successful prediction of a stock's future price could yield significant profit. The efficient-market hypothesis suggests that stock prices reflect all currently available information and any price changes that are not based on newly revealed information thus are inherently unpredictable.
In this project, we have proposed a stock market prediction model using Genetic Algorithm and Neural Networks. This technique utilises seven distinct features as the input parameters for training, and gives ‘Closing Price’ of the stock as the output.
The usage of neural networks for prediction is advantageous as they are able to learn from examples only and after their learning is finished, they are able to catch hidden and strongly non-linear dependencies, even when there is a significant noise in the training set.Genetic Algorithms are more suited for optimization problems. Hence, it is used to optimize the parameters of the Neural Network for more accurate predictions.
A programming language must be combined with special tools that support the task that has to be performed, whether one is modelling data or analysing an image. Therefore, for this project, MATLAB is used as the MATLAB toolboxes offer professionally developed, rigorously tested and fully documented functionality for scientific and engineering applications.

## 2.	Background

### 2.1	Neural Network

#### What is an Artificial Neural Network?
An Artificial Neural Network (ANN) is an information processing paradigm that is inspired by the way biological nervous systems, such as the brain, process information. The key element of this paradigm is the novel structure of the information processing system.
 It is composed of a large number of highly interconnected processing elements (neurones) working in unison to solve specific problems. ANNs, like people, learn by example. An ANN is configured for a specific application, such as pattern recognition or data classification, through a learning process. 
 
#### Basic Structure of ANNs
ANNs are composed of multiple nodes, which imitate biological neurons of human brain. The neurons are connected by links and they interact with each other. The nodes can take input data and perform simple operations on the data. The result of these operations is passed to other neurons. The output at each node is called its activation or node value.Each link is associated with weight. ANNs are capable of learning, which takes place by altering weight values. The following illustration shows a simple ANN 
 
Figure 1: Simple Neural Network

#### Feed Forward Neural Networks:
The information flow is unidirectional. A unit sends information to other unit from which it does not receive any information. There are no feedback loops. They are used in pattern generation/recognition/classification. They have fixed inputs and outputs.
 
Figure 2: Back Propogation Network

#### Back Propagation Algorithm:
It is the training or learning algorithm. It learns by example. If you submit to the algorithm the example of what you want the network to do, it changes the network’s weights so that it can produce desired output for a particular input on finishing the training.Back Propagation networks are ideal for simple Pattern Recognition and Mapping Tasks.

### 2.2 Genetic Algorithm

#### What is Genetic Algorithm?
The genetic algorithm is a method for solving both constrained and unconstrained optimization problems that is based on natural selection, the process that drives biological evolution. 
The genetic algorithm repeatedly modifies a population of individual solutions. At each step, the genetic algorithm selects individuals at random from the current population to be parents and uses them to produce the children for the next generation. Over successive generations, the population "evolves" toward an optimal solution.The genetic algorithm uses three main types of rules at each step to create the next generation from the current population:
•	Selection rules select the individuals, called parents, that contribute to the population at the next generation.
•	Crossover rules combine two parents to form children for the next generation.
•	Mutation rules apply random changes to individual parents to form children.

#### Outline of the Algorithm:
1.	The algorithm begins by creating a random initial population.
2.	The algorithm then creates a sequence of new populations. At each step, the algorithm uses the individuals in the current generation to create the next population. To create the new population, the algorithm performs the following steps:
a.	Scores each member of the current population by computing its fitness value. These values are called the raw fitness scores.
b.	Scales the raw fitness scores to convert them into a more usable range of values. These scaled values are called expectation values.
c.	Selects members, called parents, based on their expectation.
d.	Some of the individuals in the current population that have lower fitness are chosen as elite. These elite individuals are passed to the next population.
e.	Produces children from the parents. Children are produced either by making random changes to a single parent—mutation—or by combining the vector entries of a pair of parents—crossover.
f.	Replaces the current population with the children to form the next generation.
3.	The algorithm stops when one of the stopping criteria is met

#### Genetic Algorithm Terminology
•	Fitness Functions:
The fitness function is the function you want to optimize. For standard optimization algorithms, this is known as the objective function. The toolbox software tries to find the minimum of the fitness function.
•	Individuals:
An individual is any point to which you can apply the fitness function. The value of the fitness function for an individual is its score. For example, if the 
•	Populations and Generations:
A population is an array of individuals. For example, if the size of the population is 100 and the number of variables in the fitness function is 3, you represent the population by a 100-by-3 matrix. The same individual can appear more than once in the population
At each iteration, the genetic algorithm performs a series of computations on the current population to produce a new population. Each successive population is called a new generation.
•	Diversity
Diversity refers to the average distance between individuals in a population. A population has high diversity if the average distance is large; otherwise it has low diversity. 
Diversity is essential to the genetic algorithm because it enables the algorithm to search a larger region of the space.
•	Fitness Values and Best Fitness Values
The fitness value of an individual is the value of the fitness function for that individual. Because the toolbox software finds the minimum of the fitness function, the best fitness value for a population is the smallest fitness value for any individual in the population.
•	Parents and Children
To create the next generation, the genetic algorithm selects certain individuals in the current population, called parents, and uses them to create individuals in the next generation, called children. Typically, the algorithm is more likely to select parents that have better fitness values.

3. Proposed Work
3.1 Flowchart                                                                                              ----
Fig: Flowchart of Operation

### 3.2 Features
In machine learning and pattern recognition, a feature is an individual measurable property or characteristic of a phenomenon being observed.
 Choosing informative, discriminating and independent features is a crucial step for effective algorithms in pattern recognition, classification and regression. 
The features used in this project are as follows:
1.	Opening Price:The opening price is the price at which a security first trades upon the opening of an exchange on a given trading day.
2.	High Price:  Today's highis the highest price at which a stock traded during the course of the day. Today'shigh is typically higher than the closing or opening price.
3.	Low Price: Today's low is the lowest price at which a stock trades over the course of a trading day. Today's low is typically lower than the opening or closing price.
Simple moving average (SMA): The simple moving average is the most basic of the moving averages used for trading. The simple moving average formula is calculated by taking the average closing price of a stock over the last "x" periods.
4.	Simple moving average over 10 days:  This value is the average of any stock’s closing price for the last 10 days
5.	Simple moving average over 50 days: This value is the average of any stock’s closing price for the last 50 days.
6.	Exponential moving average over 10 days:  This value is the exponential average of any stock’s closing price for the last 10 days
7.	Exponential moving average over 50 days: This value is the exponential average of any stock’s closing price for the last 50 days.

The exponential moving average is a type of moving average that gives more weight to recent prices in an attempt to make it more responsive to new information. 


There are three steps to calculating an exponential moving average (EMA). First, calculate the simple moving average for the initial EMA value. An exponential moving average (EMA) has to start somewhere, so a simple moving average is used as the previous period's EMA in the first calculation.
 Second, calculate the weighting multiplier.
 Third, calculate the exponential moving average for each day between the initial EMA value and today, using the price, the multiplier, and the previous period's EMA value. 
 


### 3.3 Training data and parameter values
The data used to train the neural network is the securities exchange on the Bombay Stock Exchange (BSE) for the time period Jan 1, 1996 to Jan 1 2016.
The data used to test the neural network is from Jan 2016 to 31 July 2017.

Table 1 and Table 2 provide details about the various parameters for optimisation using the Genetic Algorithm and Neural Networks. 
Parameter Name	Value
Population Size	50
Tournament Size	2
Crossover Fraction	0.8
Migration Fraction	0.2
Migration Interval	20
Table 1: Parameters used for Genetic Algorithm

Parameter Name	Value
Training	Gradient Descent with Momentum and Adaptive Learning Rate
Performance Parameter	Mean square Error
Learning Rate	0.001
Maximum number of Epochs	8000
Table 2: Parameters used for Neural Networks
For the neural network we have used the Gradient Descent with Momentum and Adaptive Learning Rate to achieve better optimisation results as compared to simple Gradient Descent algorithm. This algorithm tunes the learning rate automatically by observing the regression trace. The number of epochs was set to 8000 to achieve best possible results for each computation.  

#### 3.4Procedure
##### Step 1: Calling the `optimtool` (Optimisation Toolbox) 
The first step involves calling the optimtool function in the Matlab command window. 
In the solver type select Genetic Algorithm. Set the lower and upper bounds of the genetic algorithm they define the number of hidden nodes in the hidden layer.
Following this click `Start`.
 
Figure 3: Optimisation Tool

##### Step 2: Creating Repetitive Neural Networks for different number of hidden nodes
The genetic algorithm selects the various values of the number of nodes in the hidden layer and calculates and compares the mean square error (M.S.E) which is the performance function. The no. of hidden nodes which correspond to the least Mean Square Error is defined as the ‘Elite’ population and is further used for computation. 
 
Figure 3: Neural Network Structure in Matlab

 
Figure 4: Training Routine of Neural Network

 
Figure 5: Optimisation Results






##### Step 3: Plotting the Final Data Plots(for over 400 points ~ 2years data)
 
Figure 6: Graphical Plot for Predicted and Actual Values









### 4.	Experimentation Results
In this section we have presented the output of the neural network for predicting the future values for training for different number of years i.e. 5, 10 and 19 years.

#### 1. 5 years
The mean square error offered in this case was 1.1902 x 107 while the number of nodes selected which corresponds to best optimisation results was 6.
 
Figure 7: Optimisation Results for 5 years
 
Figure 8: Graphical Plot for Predicted and Actual Values

#### 2. 10 years
The mean square error offered in this case was 1.132 x 105 while the number of nodes selected which corresponds to best optimisation results was 6.
 
Figure 9: Optimisation Results for 10 years
 
Figure 10: Graphical Plot for Predicted and Actual Values

#### 3. 19 years
The mean square error offered in this case was 1.0192 x 104while the number of nodes selected which corresponds to best optimisation results was 2.
 
Figure11: Optimisation Results for 10 years
 
Figure 12: Graphical Plot for Predicted and Actual Values

