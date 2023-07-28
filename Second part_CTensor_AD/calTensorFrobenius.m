
function [value2]=calTensorFrobenius(X)
[~,~,I3]=size(X);
X_FFT=bdiag(fft(X,[],3));
value2=norm(X_FFT,'f');
value2=value2*1/((I3)^0.5);
end