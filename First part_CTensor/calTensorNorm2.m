
function [val]=calTensorNorm2(A)
[~,S,~]=tensorSvd(A);
S_fft=fft(S,[],3);
val=S_fft(1,1,1);
end


function [U,S,V]=tensorSvd(B)
B=fft(B,[],3);
[I1,I2,I3]=size(B);
U=zeros(I1,I1,I3);
S=zeros(I1,I2,I3);
V=zeros(I2,I2,I3);
for i=1:I3
   [U(:,:,i),S(:,:,i),V(:,:,i)]=svd(B(:,:,i));
end
U=ifft(U,[],3);
S=ifft(S,[],3);
V=ifft(V,[],3);
end


