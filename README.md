# FE595-SKLearn-Intro

*Use a linear regression model with the Boston housing data set. 
Your code should then return which factor has the largest effect on the price of housing in Boston. 
(This is not the correlation coefficient. This is the absolute value of the slope.)*

question01.py solves this problem by creating a linear model and finding the predictor with the largest absolute value
for the slope. This predictor is NOX.

*Use a KMeans regression model with the Iris data set. Graph the fit when using differing numbers of clusters. 
Graph the result and either corroborate or refute the assumption that the data set represents 3 different varieties of iris.*

question02.py creates 4 KMeans regression models from the iris data set with 2, 3, 4, and 5 clusters. The script generates
a plot for each model. It is possible that the data set represents 3 varieties of iris, but there is significant 
overlap between the yellow and purple clusters.
