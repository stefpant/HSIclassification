function cm = find_confusion_matrix(X,Y,M)
%X:predicted labels of points
%Y:actual labels of points
%M: number of classes
%
%return the confusion matrix of X,Y

[a,N]=size(X);%N=dataset points
cm=zeros(M);
for i=1:N
    %row:classes
    %col:predictions
    cm(Y(1,i),X(1,i))=cm(Y(1,i),X(1,i)) + 1;
end

end

