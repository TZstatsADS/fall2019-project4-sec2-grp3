# Project 4: Algorithm implementation and evaluation: Collaborative Filtering

### [Project Description](doc/project4_desc.md)

Term: Fall 2019

+ Team 3
+ Projec title: Collaborative Filtering （A1 vs A2 given P3)


+ Team members
	+ team member 1: Chen, Kanyan kc3207@columbia.edu
	+ team member 2: Fang, Dingyi df2709@columbia.edu
	+ team member 3: Gong, Yuhan yg2622@columbia.edu
	+ team member 4: Gu, Feichi fg2458@columbia.edu
	+ team member 5: Zhang, Haoyu hz2563@columbia.edu
	
	
+ Project summary: 
Nowadays we are constantly being recommended from various sources, such as suitable movies and popular music. These recommendations focus on a more personal level than years ago. The major work of this project is to investigate, implement, evaluate and compare matrix factorization techniques which create latent features vectors for both items and users based on item rating patterns.

Specifically, we compare two factorization algorithms:【Stochastic Gradient Descent】vs【Gradient Descent with Probabilistic Assumptions】. The main measure we use is RMSE. By cross validation, we tune parameters such as the dimension of factor and the penalty parameter lambda. In addition, to improve the performance of factorization, we use【kernel ridge regression】to postprocess previous optimal SVD results. Here we mainly adopt rbf kernel and tune its parameters such as alpha and gamma.


+ Main result: 


+ Contribution statement
	+ Chen, Kanyan: 
	+ Fang, Dingyi: 
	+ Gong, Yuhan: 
	+ Gu, Feichi:
	+ Zhang, Haoyu:


+ Reference:
1. Yehuda Koren, Robert Bell, Chris Volinsky. Matrix Factorization Techniques For Recommender Systems. NIPS'07 Proceedings of the 20th International Conference on Neural Information Processing Systems, December 03 - 06, 2007.
2. Ruslan Salakhutdinov, Andriy Mnih. Probabilistic Matrix Factorization. Journal Computer Volume 42 Issue 8 Pages 30-37, August 2009. 
3. Arkadiusz Paterek. Improving Regularized Singular Value Decomposition for Collaborative Filtering. January 2007

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
