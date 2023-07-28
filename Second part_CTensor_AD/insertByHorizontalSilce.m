
function [tensorSketch,indexZero,projectionTensor,rate]=insertByHorizontalSilce(tensorSketch,tensorSketch_I1,insertSlice,shrinkLevel,indexZero,topK,projectionTensor)
tensorSketch(indexZero,:,:)=insertSlice;
indexZero=indexZero+1;
rate=0;
if(indexZero>tensorSketch_I1)
    [~,S,V]=tensorSvd(tensorSketch);
    [newS,~]=shrinkOperationRow(S,shrinkLevel);
    [rate]=calLamda(S,topK);
    tensorSketch=tprod(newS,tran(V)); 
    projectionTensor=tprod(V(:,1:topK,:),tran(V(:,1:topK,:)));
    indexZero=shrinkLevel;    
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












