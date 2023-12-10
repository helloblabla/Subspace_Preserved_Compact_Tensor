function [U,S,V,B_k]=tensorSvds(B,k)

B=fft(B,[],3);
[I1,I2,I3]=size(B);
U=zeros(I1,k,I3);
S=zeros(k,k,I3);
V=zeros(I2,k,I3);
for i=1:I3
   [U(:,:,i),S(:,:,i),V(:,:,i)]=svds(B(:,:,i),k);
end
U=ifft(U,[],3);
S=ifft(S,[],3);
V=ifft(V,[],3);
B_k=tprod(tprod(U,S),tran(V));
end