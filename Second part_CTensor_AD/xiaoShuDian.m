
function [need_str]=xiaoShuDian(a,n)
b=a*10^n;
aa=round(b);
need_num=aa/10^n; 
need_str=string(num2str(need_num));
end
