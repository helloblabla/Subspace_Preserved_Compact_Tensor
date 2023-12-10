
function [value1]=calTensorFrobenius(X)

[I1,I2,I3]=size(X);
value1=0;
for i=1:I1
    for j=1:I2
        for k=1:I3
            value1=value1+X(i,j,k)^2;
        end
    end
end
value1=sqrt(value1);
end