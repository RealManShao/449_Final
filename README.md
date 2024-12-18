# 449_Final
## Pre-request
Pytorch 2.5

Enron Spam Datasets http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html 

[V. Metsis, I. Androutsopoulos and G. Paliouras, "Spam Filtering with 
Naive Bayes - Which Naive Bayes?". Proceedings of the 3rd Conference 
on Email and Anti-Spam (CEAS 2006), Mountain View, CA, USA, 2006.]
## Result
I Use 2 dataset to train and 2 dataset to test.

Since I use Windows with the AMD 7900XT, I can not use ROCm, and the code runs on the CPU (AMD 5900X). So both training and testing dataset is small, I will deploy it on Linux later and try some bigger dataset to see the performance.
