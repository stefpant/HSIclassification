function [pred] = knn_predict(xtrain,ytrain,xtest,k)
%predict classes for xtest points
%with knn classifier
    pred=[];
    [l,N]=size(xtest);
    V=size(xtrain,2);
    for i=1:N %for every test point
        dist=[];
        for j=1:V%find i's distance with every train point
            dist=[dist; find_dist(xtrain(:,j),xtest(:,i)) ytrain(j)];
        end
        arr=sortrows(dist,1);%sort dists
        classes=[];
        for j=1:k
            classes=[classes arr(j,2)];%get classes from first k points
        end
        pred=[pred mode(classes)];%majority voting to find i's class
    end
end
