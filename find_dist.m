function [d] = find_dist(a,b)
%euclidean distance between 2 points a,b
    n=size(a,1);
    tempd=0;
    for i=1:n
        tempd=tempd+(a(i)-b(i))^2;
    end
    d=sqrt(tempd);
end

