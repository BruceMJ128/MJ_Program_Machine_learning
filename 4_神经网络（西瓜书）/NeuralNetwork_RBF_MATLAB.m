
clear
x=[0 0;0 1;1 0;1 1];
y=[0;1;1;0];


t=10;   %隐层神经元数目 大于输入层
p=rand(4,t);    %径向基函数的值  
ty=rand(4,1);   %输出值
w=rand(1,t);    %隐层第i个神经元与输出神经元的权值
b=rand(1,t);    %样本与第i个神经元的中心的距离的缩放系数
tk=0.5;

%[id,c]=kmeans(x,2);
c = rand(t,2);  %隐层第i个神经元的中心

kn=0;       %来及迭代次数
sn=0;           %同样的累计误差值累积次数
old_ey=0;       %前一次迭代的累计误差

while(1)
    kn=kn+1;
    %计算每个样本的径向基函数值
    for i=1:4
       for j=1:t
          p(i,j)=exp(-b(j)*(x(i,:)-c(j,:))*(x(i,:)-c(j,:))') ;
       end
       ty(i)=w* p(i,:)';
    end
    %计算累计误差
    ey =  (ty-y)'*(ty-y);

    %更新w,b
    dw=zeros(1,t);
    db=zeros(1,t);
    for i=1:4
        for j=1:t
            dw(j)=dw(j)+(ty(i)-y(i))* p(i,j);
            db(j)=db(j)-(ty(i)-y(i))*w(j)*(x(i,:)-c(j,:))*(x(i,:)-c(j,:))'*p(i,j);
        end
    end

    w = w - tk *  dw / 4;
    b = b - tk *  db / 4;
    %迭代终止条件
   if(abs(old_ey-ey)<0.0001)
       sn=sn+1;
       if(sn==10)
           break;
       end
   else
       old_ey=ey;
       sn=0;
   end

end
