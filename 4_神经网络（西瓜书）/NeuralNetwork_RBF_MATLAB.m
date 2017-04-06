
clear
x=[0 0;0 1;1 0;1 1];
y=[0;1;1;0];


t=10;   %������Ԫ��Ŀ ���������
p=rand(4,t);    %�����������ֵ  
ty=rand(4,1);   %���ֵ
w=rand(1,t);    %�����i����Ԫ�������Ԫ��Ȩֵ
b=rand(1,t);    %�������i����Ԫ�����ĵľ��������ϵ��
tk=0.5;

%[id,c]=kmeans(x,2);
c = rand(t,2);  %�����i����Ԫ������

kn=0;       %������������
sn=0;           %ͬ�����ۼ����ֵ�ۻ�����
old_ey=0;       %ǰһ�ε������ۼ����

while(1)
    kn=kn+1;
    %����ÿ�������ľ��������ֵ
    for i=1:4
       for j=1:t
          p(i,j)=exp(-b(j)*(x(i,:)-c(j,:))*(x(i,:)-c(j,:))') ;
       end
       ty(i)=w* p(i,:)';
    end
    %�����ۼ����
    ey =  (ty-y)'*(ty-y);

    %����w,b
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
    %������ֹ����
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
