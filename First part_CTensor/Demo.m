%Demo
clear;
path='..\Data\syntheticData_1.mat ';
load(path);
Y1=syntheticData_1;
path='..\Data\syntheticData_2.mat ';
load(path);
Y2=syntheticData_2;
Y=zeros(10000,50,50);
Y(1:5000,:,:)=Y1;
Y(5001:10000,:,:)=Y2;
A=double(Y);
[n,I2,I3]=size(A);
LArr=[500];
[~,LArrLength]=size(LArr);
shrinkLevelArr=[30];
[~,shrinkLevelArrLength]=size(shrinkLevelArr);
topKArr=[49];
[~,topKArrLength]=size(topKArr);
for h=1:LArrLength
    ID=randperm(10000000,1);
    L=LArr(1,h);
    B=zeros(L,I2,I3);
    for i=1:shrinkLevelArrLength
        shrinkLevel=shrinkLevelArr(1,i);
        for j=1:topKArrLength
            topK=topKArr(1,j);
            [covError,proError,rateArr,max_shrinkVal,sumTube_shrinkVal,countShrink,proErrorUpBound]=steamTensorSketch(A,B,topK,shrinkLevel);
            disp("topK="+topK+",shrinkLevel="+shrinkLevel+",L="+L+",covError="+covError+",proError="+proError+",sum(max_shrinkVal)="+sum(max_shrinkVal)+",sum(sumTube_shrinkVal)="+sum(sumTube_shrinkVal)+",countShrink="+countShrink+",proErrorUpBound="+proErrorUpBound+",ID="+ID);
        end
    end   
    str=join(string(rateArr),',');
end




function [covError,proError,rateArr,max_shrinkVal,sumTube_shrinkVal,countShrink,proErrorUpBound]=steamTensorSketch(A,B,topK,shrinkLevel)
[n,~,~]=size(A);
[L,~,~]=size(B);
max_shrinkVal=zeros(1,n);
sumTube_shrinkVal=zeros(1,n);
countShrink=0;
indexZore=1;
for i=1:n
    B(indexZore,:,:)=A(i,:,:);
    indexZore=indexZore+1;
    if(indexZore>L)
         [U,S,V]=tensorSvd(B);
          [newS,shrinkVal]=shrinkOperationRow(S,shrinkLevel);
          countShrink=countShrink+1;
          max_shrinkVal(1,countShrink)=max(shrinkVal);
          sumTube_shrinkVal(1,countShrink)=sumTube(shrinkVal);
          [rateArr]=calLamda(S);
          B=tprod(newS,tran(V)); 
          V_k=V(:,1:topK,:);
          indexZore=shrinkLevel;         
    end
    lastB=B;
end
expression1=tprod(tran(A),A);
expression2=tprod(tran(B),B);
covError=calTensorNorm2(expression1-expression2)/calTensorNorm2(expression1);

expression3=A-tprod(tprod(A,V_k),tran(V_k));
proError=calTensorFrobenius(expression3)^2/(calTensorFrobenius(A)^2);

%||A-A_k||_F^2+k*max_shrinkVal
[~,~,~,A_k]=tensorSvds(A,topK);
proErrorUpBound=calTensorFrobenius(A-A_k)^2+topK*sum(max_shrinkVal);
end











function [rateArr]=calLamda(S_tensor)
S=S_tensor(:,:,1);
[I1,I2]=size(S);
minI=min(I1,I2);
rateArr=zeros(1,minI);
Svalue=0;
for i=1:minI
    Svalue=Svalue+S(i,i);
    if(S(i,i)<0)
        errID = 'myComponent:inputError';
        msgtext = 'S(i,i)ÓÐ¸ºÊý';
        ME = MException(errID,msgtext);
        throw(ME);
    end
end
hasSValue=0;
for i=1:minI
    hasSValue=hasSValue+S(i,i);
    rateArr(1,i)=hasSValue/Svalue;
end
end


function [newS,shrinkVal]=shrinkOperationRow(S,shrinkLevel)
[I1,I2,I3]=size(S);
S_FFT=fft(S,[],3);
S_FFT_square=zeros(I1,I2,I3);
cut=zeros(I1,I2,I3);
newS=zeros(I1,I2,I3);
minI=min(I1,I2);
shrinkVal=zeros(1,1,I3);
for i=1:I3
    for j=1:minI
        S_FFT_square(j,j,i)=S_FFT(j,j,i)^2;
    end
    shrinkVal(1,1,i)=S_FFT_square(shrinkLevel,shrinkLevel,i);
    for j=1:minI
        cut(j,j,i)=shrinkVal(1,1,i);
    end
end
for i=1:I3
    temp=max(S_FFT_square(:,:,i)-cut(:,:,i),0);
    newS(:,:,i)=sqrt(temp);
end
newS=ifft(newS,[],3);
end








function[sumValue]=sumTube(X)
sumValue=0;
[~,~,I3]=size(X);
for i=1:I3
    sumValue=sumValue+X(1,1,i);
end
end




