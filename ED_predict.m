function y_pred = ED_predict(X,m)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    y_pred=[];
    [l,N]=size(X);
    for i=1:N %for every point
        k_predictions=[];
        for k=1:5 %for every class
            sumprob=0;
            for j=1:l %for every feature
                sumprob=sumprob + (X(j,i)-m(k,j))^2;
            end
            k_predictions=[k_predictions sumprob];%should be 'sumprob^0.5' but doesn't change anything
        end
        [M,I]=min(k_predictions);
        y_pred=[y_pred I];
    end
end

