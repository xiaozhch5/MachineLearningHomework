function f=bayesian_fun(t2,t1,W1,W2,W3,w1,w2,w3,w10,w20,w30,p)

x=[t1,t2]';

f=x'*W1*x+w1'*x+w10 - p(1)*(x'*W2*x+w2'*x+w20) - p(2)*(x'*W3*x+w3'*x+w30);
end