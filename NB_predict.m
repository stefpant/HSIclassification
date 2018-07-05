function ypred = NB_predict(X,m,s)
%predict the class of every one of the N points of X dataset(lxN)
%for every x in X:
%   find all 5 P(x) = P(x|w_i)
%   assign point x in class i with max P(x)
%   P(x|w_i)=p(x1,i)*...*p(xl,i)-->
%   actually finds the log[P(x|w_i)]=log[p(x1,i)]+...+log[p(xl,i)]
%   and log[p(x1,i)] = -log((2pi)^0.5*s) - 0.5*((x-m)/s)^2
    ypred=[];
    [l,N]=size(X);
    for i=1:N %for every point
        k_predictions=[];
        for k=1:5 %for every class
            prob_prod=0;
            for j=1:l %for every feature
                prob= log(((2*pi)^(0.5))*s(k,j)) + 0.5*(((X(j,i)-m(k,j))/s(k,j))^2);
                prob_prod=prob_prod - prob;
            end
            k_predictions=[k_predictions prob_prod];
        end
        %k_predictions
        [M,I]=max(k_predictions);
        ypred=[ypred I];
    end
end

