function [bestk,bestacc] = knn_cross_val(x,y,setk)
%given a dataset x and its class labels
%find the best k in setk and accuracy with cross validation
%for knn classifier
%(stratified 5-fold cross val)

N=size(x,2);%N data points
index1=[];
index2=[];
index3=[];
index4=[];
index5=[];
for i=1:N %seperate indexes for labels in x
   if(y(i)==1)%to get point i-->x(:,i)
       index1=[index1 i];
   elseif(y(i)==2)
       index2=[index2 i];
   elseif(y(i)==3)
       index3=[index3 i];
   elseif(y(i)==4)
       index4=[index4 i];
   elseif(y(i)==5)
       index5=[index5 i];
   end
end
index = {index1;index2;index3;index4;index5};
%index{1}=index1;
%index{2}=index2;
%index{3}=index3;
%index{4}=index4;
%index{5}=index5;
a = [get_indices(index1,5); get_indices(index2,5); get_indices(index3,5); get_indices(index4,5); get_indices(index5,5)];
ks=size(setk,2);%9 dif k(1 3 5...),9 times we'll run real cross val
bestacc=0;
for j=1:ks
   k=setk(j);
   accs=0;
   for i=1:5
       traini=[];
       testi=[];
       for cl=1:5%classes
           if(i<2)
               num=0;%from num-a(cl,i) in test
           else
               num=sum(a(cl,1:(i-1)));
               traini=[traini index{cl}(1:num)];
           end
           testnum=num+a(cl,i);
               testi=[testi index{cl}((num+1):testnum)];
           if(i<5)
               traini=[traini index{cl}((testnum+1):end)];
           end
       end%now traini and testi have the x indexes to use in cross val
       trainXi2=sort(traini);
       testXi2=sort(testi);
       testXi=[];
       testYi=[];
       for z=1:size(testXi2,2)
           testXi=[testXi x(:,testXi2(z))];
           testYi=[testYi y(testXi2(z))];
       end
       trainXi=[];
       trainYi=[];
       for z=1:size(trainXi2,2)
           trainXi=[trainXi x(:,trainXi2(z))];
           trainYi=[trainYi y(trainXi2(z))];
       end
       y_pred=knn_predict(trainXi,trainYi,testXi,k);
       %then check testYi and y_pred for accuracy
       counter=0;
       for z=1:size(testYi,2)
           if(testYi(z)==y_pred(z))
               counter=counter+1;
           end
       end
       accs=accs + counter/size(testYi,2);
   end
   accuracy=accs/5;
   if(accuracy>bestacc)
           bestacc=accuracy;
           bestk=k;
   end
end
end

