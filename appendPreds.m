function [x] = appendPreds(img,preds,response,pos)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    x=img;
    N=size(preds,2);
    for i=1:N
        if(preds(i)==response(i))
            x(pos(i,1),pos(i,2))=preds(i);
        end
    end
end

