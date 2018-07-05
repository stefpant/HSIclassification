function a = get_indices(x,n)
%ret an array 1xN of num of indexes
%to use as test in cross val
k=size(x,2);
a=[];
for i=1:n
    tempk=floor(k/(n+1-i));
    a=[a tempk];
    k=k-tempk;
end
end

