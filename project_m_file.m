% This is a supporting MATLAB file for the project

clear
format compact
close all

load Salinas_hyperspectral %Load the Salinas hypercube called "Salinas_Image"
[p,n,l]=size(Salinas_Image) % p,n define the spatial resolution of the image, while l is the number of bands (number of features for each pixel)

load classification_labels 
% This file contains three arrays of dimension 22500x1 each, called
% "Training_Set", "Test_Set" and "Operational_Set". In order to bring them
% in an 150x150 image format we use the command "reshape" as follows:
Training_Set_Image=reshape(Training_Set, p,n); % In our case p=n=150 (spatial dimensions of the Salinas image).
Test_Set_Image=reshape(Test_Set, p,n);
Operational_Set_Image=reshape(Operational_Set, p,n);

%Depicting the various bands of the Salinas image
for i=1:l
    figure(1), imagesc(Salinas_Image(:,:,i))
    %pause(0.05) % This command freezes figure(1) for 0.05sec. 
end

% Depicting the training, test and operational sets of pixels (for the
% pixels depicted with a dark blue color, the class label is not known.
% Each one of the other colors in the following figures indicate a class).
figure('Name','Training set'), imagesc(Training_Set_Image)
figure('Name','Test set'), imagesc(Test_Set_Image)
figure('Name','Operation set'), imagesc(Operational_Set_Image)

% Constructing the 204xN array whose columns are the vectors corresponding to the
% N vectors (pixels) of the training set (similar codes cane be used for
% the test and the operational sets).
Train=zeros(p,n,l); % This is a 3-dim array, which will contain nonzero values only for the training pixels
for i=1:l
     %Multiply elementwise each band of the Salinas_Image with the mask 
     % "Training_Set_Image>0", which identifies only the training vectors.
    Train(:,:,i)=Salinas_Image(:,:,i).*(Training_Set_Image>0);
    figure(5), imagesc(Train(:,:,i)) % Depict the training set per band
    %pause(0.05)
end

Train_array=[]; %This is the wanted 204xN array
Train_array_response=[]; % This vector keeps the label of each of the training pixels
Train_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.
for i=1:p
    for j=1:n
        if(Training_Set_Image(i,j)>0) %Check if the (i,j) pixel is a training pixel
            Train_array=[Train_array squeeze(Train(i,j,:))];
            Train_array_response=[Train_array_response Training_Set_Image(i,j)];
            Train_array_pos=[Train_array_pos; i j];
        end
    end
end

N=size(Train_array,2);%N=num of active pixels in image(their class>0)
Train_c1=[];
Train_c2=[];
Train_c3=[];
Train_c4=[];
Train_c5=[];
for i=1:N %seperate the N 'active' pixels to classes
   if(Train_array_response(i)==1)
       Train_c1=[Train_c1 Train_array(:,i)];
   elseif(Train_array_response(i)==2)
       Train_c2=[Train_c2 Train_array(:,i)];
   elseif(Train_array_response(i)==3)
       Train_c3=[Train_c3 Train_array(:,i)];
   elseif(Train_array_response(i)==4)
       Train_c4=[Train_c4 Train_array(:,i)];
   elseif(Train_array_response(i)==5)
       Train_c5=[Train_c5 Train_array(:,i)];
   end
end

%let's find the parameters for every class m,s (5xl)
%where m[i]=[m1,m2,..,ml],s[i]=[s1,s2,..,sl] and 1<=i<=5 for every class
m=[]; %used also for ED classifier
s=[];

[temp_m1,temp_s1]= findNB_params(Train_c1);
[temp_m2,temp_s2]= findNB_params(Train_c2);
[temp_m3,temp_s3]= findNB_params(Train_c3);
[temp_m4,temp_s4]= findNB_params(Train_c4);
[temp_m5,temp_s5]= findNB_params(Train_c5);

m=[temp_m1; temp_m2; temp_m3; temp_m4; temp_m5];
s=[temp_s1; temp_s2; temp_s3; temp_s4; temp_s5];
%NaiveBayes and EuclideanDistance classifiers ready for predictions

%creating X,y for test_set
Test=zeros(p,n,l); % This is a 3-dim array, which will contain nonzero values only for the training pixels
for i=1:l
     %Multiply elementwise each band of the Salinas_Image with the mask 
     % "Test_Set_Image>0", which identifies only the training vectors.
    Test(:,:,i)=Salinas_Image(:,:,i).*(Test_Set_Image>0);
end

Test_array=[]; %(X)This is the wanted 204xN array
Test_array_response=[]; % (y) This vector keeps the label of each of the training pixels
Test_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.
for i=1:p
    for j=1:n
        if(Test_Set_Image(i,j)>0) %Check if the (i,j) pixel is a training pixel
            Test_array=[Test_array squeeze(Test(i,j,:))];
            Test_array_response=[Test_array_response Test_Set_Image(i,j)];
            Test_array_pos=[Test_array_pos; i j];
        end
    end
end

imageNB=Training_Set_Image;%init the 3 images with training set to
imageED=Training_Set_Image;%visualize clf predictions
imageKNN=Training_Set_Image;

setk=1:2:17;%values for k to test knn in cross val
[bestk,acc]=knn_cross_val(Train_array,Train_array_response,setk);
y_pred=[];

%predict test set with nb classifier...++append result in y_pred array
y_pred=[y_pred; NB_predict(Test_array,m,s)];
imageNB=appendPreds(imageNB,y_pred(1,:),Test_array_response,Test_array_pos);

%predict test with ed clf ++ append result in y_pred array
y_pred=[y_pred; ED_predict(Test_array,m)];
imageED=appendPreds(imageED,y_pred(2,:),Test_array_response,Test_array_pos);

%predict test with knn using as k the bestk found by cross val ++results in y_pred
y_pred =[y_pred; knn_predict(Train_array,Train_array_response,Test_array,bestk)];
imageKNN=appendPreds(imageKNN,y_pred(3,:),Test_array_response,Test_array_pos);

for j=1:3 %create and print confusion matrices
    conM=find_confusion_matrix(y_pred(j,:),Test_array_response,5)
    pred=0;%computing accuracy
    for i=1:5
        pred=pred + conM(i,i);
    end
    pred=pred/size(Test_array,2) %<--final accuracy
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%creating X,y for operational_set
Operation=zeros(p,n,l); % This is a 3-dim array, which will contain nonzero values only for the training pixels
for i=1:l
     %Multiply elementwise each band of the Salinas_Image with the mask 
     % "Operational_Set_Image>0", which identifies only the training vectors.
    Operation(:,:,i)=Salinas_Image(:,:,i).*(Operational_Set_Image>0);
end

Operation_array=[]; %(X)This is the wanted 204xN array
Operation_array_response=[]; % (y) This vector keeps the label of each of the training pixels
Operation_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.
for i=1:p
    for j=1:n
        if(Operational_Set_Image(i,j)>0) %Check if the (i,j) pixel is a training pixel
            Operation_array=[Operation_array squeeze(Operation(i,j,:))];
            Operation_array_response=[Operation_array_response Operational_Set_Image(i,j)];
            Operation_array_pos=[Operation_array_pos; i j];
        end
    end
end

%same predictions for operation_array and print accuracy
%then printing confusion matrices
y_pred=[];
y_pred=[y_pred; NB_predict(Operation_array,m,s)];
imageNB=appendPreds(imageNB,y_pred(1,:),Operation_array_response,Operation_array_pos);

y_pred=[y_pred; ED_predict(Operation_array,m)];
imageED=appendPreds(imageED,y_pred(2,:),Operation_array_response,Operation_array_pos);

y_pred =[y_pred; knn_predict(Train_array,Train_array_response,Operation_array,bestk)];
imageKNN=appendPreds(imageKNN,y_pred(3,:),Operation_array_response,Operation_array_pos);

for j=1:3
    conM=find_confusion_matrix(y_pred(j,:),Operation_array_response,5)
    pred=0;%computing accuracy
    for i=1:5
        pred=pred + conM(i,i);
    end
    pred=pred/size(Operation_array,2) %<--final accuracy
end

totalp_image=Training_Set_Image;%init with train image
for i=1:p
    for j=1:n
        if(totalp_image(i,j)==0)
            totalp_image(i,j)=Test_Set_Image(i,j);
            if(totalp_image(i,j)==0)%if still 0 try with operational image...
                totalp_image(i,j)=Operational_Set_Image(i,j);
            end
        end
    end
end
figure('Name','Visualizing all sets'), imagesc(totalp_image)
figure('Name','Naive Bayes clf'), imagesc(imageNB)
figure('Name','Euclidean Distance clf'), imagesc(imageED)
figure('Name','K-Nearest Neighbor clf'), imagesc(imageKNN)