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

Nowadays we are constantly being recommended from various sources, such as suitable movies and popular music. These recommendations focus on a more personal level than years ago. The major work of this project is to implement, evaluate and compare matrix factorization techniques which create latent features vectors for both items and users based on item rating patterns.

Specifically, we compare two factorization algorithms: A1【Stochastic Gradient Descent】vs A2【Gradient Descent with Probabilistic Assumptions】. The main measure we use is RMSE. By cross validation, we tune parameters such as the dimension of factor and the penalty parameter lambda. In addition, to improve the performance of factorization, we use P3【kernel ridge regression】to postprocess previous optimal SVD results. Here we mainly adopt rbf kernel and tune its parameters such as alpha and gamma.


+ Main result: 

A1 has lower test error and takes shorter time to process factorization than A2. A1 is more time-saving because A1 converges quickly and we set epoch value for A1 to be 200 while epoch for A2 is 500. In addition, P3 postprocessing seems to lose its power regarding improving accuracy on both algorithms. We think this may be reasonable because kernel ridge regression only utilize information in movie feature vectors and thus in some cases may not be a wise choice to improve performance of our recommender system.


+ Contribution statement: Kanyan Chen is the project leader. Other members contributed equally. All team members approve our work presented in this GitHub repository including this contributions statement.
	+ Chen, Kanyan: create Python scripts for two factorization algorithms, kernel ridge regression, and cross validation; finalize report with Yuhan; offer help for each step
	+ Fang, Dingyi: utilize cross validation to tune parameters for A2 (Gradient Descent with Probabilistic Assumptions)
	+ Gong, Yuhan: arrange group meetings, manage Github, calculate running time and plotting, edit final report with Kanyan and make presentation
	+ Gu, Feichi: tune parameters for rbf kernel ridge regression, try different kernels for P3
	+ Zhang, Haoyu: utilize cross validation to tune parameters for A1 (Stochastic Gradient Descent)


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
