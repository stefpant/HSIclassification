function [m,s] = findNB_params(X)
%given a dataset X (lxN)
%where N are the points and l the features
%estimate the l parameters of means m and s
    m=[];
    s=[];
    [l,N]=size(X);
    for i=1:l
        temp_m=sum(X(i,:))/N;
        m=[m temp_m];
        ssum=0;
        for j=1:N
            ssum=ssum+(X(i,j)-temp_m)^2;
        end
        s=[s (ssum/N)^0.5];
    end
end

