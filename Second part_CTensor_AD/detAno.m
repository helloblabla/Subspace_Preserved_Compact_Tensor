

function [curScore]=detAno(testSlice,projectionTensor)
lateralSlice=tran(testSlice);
expression=(lateralSlice-tprod(projectionTensor,lateralSlice));
curScore=calTensorFrobenius(expression);
end