# Problem description
设x为一个d维的二值向量（即其分量取值为0或1），服从多维伯努利分布

<center><img src="http://chart.googleapis.com/chart?cht=tx&chl= \begin{equation}p(x_i|\Theta) = \prod_{i=1}^d \Theta_i^{x_i}(1-\Theta_i)^{1-x_i} \end{equation}" style="border:none;"></center>

其中<img src="http://chart.googleapis.com/chart?cht=tx&chl= \Theta = (\Theta_1, \cdots, \Theta_d)^T" style="border:none;">是未知量， 而<img src="http://chart.googleapis.com/chart?cht=tx&chl= \Theta_i" style="border:none;">为<img src="http://chart.googleapis.com/chart?cht=tx&chl= x_i=1" style="border:none;">的概率。证明，对于<img src="http://chart.googleapis.com/chart?cht=tx&chl= \Theta" style="border:none;">的最大似然估计为<img src="http://chart.googleapis.com/chart?cht=tx&chl= \begin{equation} \hat{\Theta} = \frac{1}{n}\sum\limits_{k=1}^nX_k \end{equation}" style="border:none;">
