%EE505- Neural Networks and Applications
%Project 1- OCR 
%March 2014
%NURAY GUL-130505009
%nuray.gul@atilim.edu.tr
%------------------%
%This prog is based on training of a layer of neurons by using Perceptron
%There are 10 neurons and 10 outputs
clear all 
clc
%Defining training set
net = newp([8 24; 4 17 ],20);
Class1=[[16 4]', [11 6]', [13 8]', [21 7]',[16 6]', [19 9]',[21 5]',[26 6]',[22 10]',[24 8]'];
Class2=[[11 15]', [12 16]', [10 14]', [9 17]',[13 11]', [8 12]',[14 10]',[12 12]',[11 14]',[10 13]'];
plot(Class1(1,:),Class1(2,:),'ro','LineWidth',3)
hold on;
plot(Class2(1,:),Class2(2,:),'b+','LineWidth',3)
grid on;
title('Perceptron Algorithm');
%% Solution
X=[[Class1',ones(10,1)];[Class2',ones(10,1)]]'%Data Matrix
P=[Class1, Class2]  %%   2-by-20 training data
bias=ones(20,1)
%---Defining Weights---%
w=rand(20,2)  %random weights 20-by-2
net.IW{1,1}=w %set random 10-by-35 weights to network

%---Defining Target---%
t1=[ones(1,10);zeros(19,10)];
t2=[zeros(19,10);ones(1,10)];
t= [t1 t2]%target 20-by-2 matrix of ones 
           %target matrix must contain values of either 0 or 1,as
           %perceptrons(with hardlim transfer functions)can only output such values

Learning_rate=0.2; %Learning rate parameter

%------ Training phase------%
for a=1:100 % 100 is accepted as iteration times
  
   v=w*P; %10-by-10 transfer function without bias
   % Defining Hard-Limit Transfer Function for Perceptron
for i=1:20; 
    for j=1:20
        if v(i,j)>0 %activation function
            y(i,j)=1; %output function
        else
            y(i,j)=0;
        end
    end
end

e=t-y; %error 
dw=Learning_rate*e*P'; %changing amount at weights
w=w+dw; % new weights after updating

end

    otput=y %output after trainng
    error=e %Error after training
    hold on
    plot(w(1,:),w(2,:))