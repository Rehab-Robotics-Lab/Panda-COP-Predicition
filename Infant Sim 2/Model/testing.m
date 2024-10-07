arm=readtable('larm_6.csv');
dt=0.14;
dt_min=dt/60;

t=arm.time/1000;
mean(diff(t));

% anglef=medfilt1(arm.arm1Angle,5);
% angle=arm.arm1Angle;
% load=arm.arm1Load;
% 
% dt=diff(t);
% find(dt>0.045);
% tbool=dt>0.045;
% % stops=t(tbool);
% 
% plot(t,load)
% hold on
% plot(t(tbool),load(tbool),'.')


pwm=arm.position;
angle=deg2rad(((pwm-1835)/4095)*360);
d_angle=gradient(angle,dt);
dd_angle=gradient(d_angle,dt)*573;

spd=arm.velocity*.229;
accel=gradient(spd,dt_min);
hold on
plot(t,accel)
% hold on
plot(t,dd_angle)
legend('Grnd Trth','Grad')
% plot(t(tbool),load(tbool),'.')

%% sk
%%parameters
%mass
m1=.377;
m2=.377;
m3=.377;
m4=.3;
%length
l0=.1;
l1=.1;
l2=.1;
le=.08;
%
r3=.02;
r4=.015;
%intertia
I1=I(l2,m3,r3);
I2=I(l2,m3,r3);
I3=I(l2,m3,r3);
I4=I(le,m4,r4);

arm=readtable('larm_8.csv');

[n,b]=size(arm);

range=1:n;

arm_ang=(((arm.position-1835)/4095)*360)-45;
% arm_ang = filtfilt(b,a,arm_ang);

theta2=transpose(deg2rad(arm_ang(range)));
theta1=zeros(length(theta2));
t3=0;
theta3=deg2rad(ones(length(theta2))*t3);
t4=70;
theta4=deg2rad(ones(length(theta2))*t4);


dt=0.14;

%theta dot
theta_d1=gradient(theta1,dt);
theta_d2=gradient(theta2,dt);
theta_d3=gradient(theta3,dt);
theta_d4=gradient(theta4,dt);
%theta double dot
theta_dd1=gradient(theta_d1,dt);
theta_dd2=gradient(theta_d2,dt);
theta_dd3=gradient(theta_d3,dt);
theta_dd4=gradient(theta_d4,dt);


wo=[0;0;0];
vo_d=[0;0;-9.81];
wo_d=[0;0;0];

zo=[0;0;1];

f4=[0;0;0];
n4=[0;0;0];

T=zeros('like',theta2);
check=zeros('like',theta2);

for i=1:length(theta2)
% for i=n
    thet1=theta1(i);
    thet2=theta2(i);
    thet3=theta3(i);
    thet4=theta4(i);
    %theta dot
    thet_d1=theta_d1(i);
    thet_d2=theta_d2(i);
    thet_d3=theta_d3(i);
    thet_d4=theta_d4(i);
    %theta double dot
    thet_dd1=theta_dd1(i);
    thet_dd2=theta_dd2(i);
    thet_dd3=theta_dd3(i);
    thet_dd4=theta_dd4(i);
    
    %simple calculation
    lev1=l2/2*cos(thet2);
    lev2=((l2*cos(thet2))+(le/2*cos(thet2+thet4)));
    
    iner1=I3(2,2);
    iner2=I4(2,2)+(m4*(.12^2));

    % check(i)=(9.81*(m3*lev1+m4*lev2))-((iner1+iner2)*thet_dd2);
    check(i)=((m3*lev1^2+m4*(lev1^2+2*lev2*lev1*cos(thet4)+lev2^2))*thet_dd2)+((m3+m4)*lev1*9.81*cos(thet2))+(m4*9.81*lev2*cos(thet2+thet4));

    %T,R and P  
    T01 = TI(0 , pi/2, 0, thet1+pi/2);
    R01=T01(1:3,1:3);
    
    T12 = TI(0, pi/2 , 0, thet2+(3*pi/2));
    R12=T12(1:3,1:3);
    
    T23=TI(0, -pi/2, 0, pi+thet3);
    R23=T23(1:3,1:3);
    
    T34=TI(l2, pi/2 , 0, thet4);
    P34=T34(1:3,4);
    R34=T34(1:3,1:3);

    T45=TI(le, 0 , 0, 0);
    P45=T45(1:3,4);
    R45=T34(1:3,1:3);

    %%forward
    %FRAME1
    w1=zo*thet_d1;
    w2=transpose(R12)*w1+zo*thet_d2;
    w3=transpose(R23)*w2+zo*thet_d3;
    w4=transpose(R34)*w3+zo*thet_d4;


    w1_d=zo*thet_dd1;
    w2_d=(transpose(R12)*w1_d)+cross(transpose(R12)*w1,zo*thet_d2)+(zo*thet_dd2);
    w3_d=(transpose(R23)*w2_d)+cross(transpose(R23)*w2,zo*thet_d3)+(zo*thet_dd3);
    w4_d=(transpose(R34)*w3_d)+cross(transpose(R34)*w3,zo*thet_d4)+(zo*thet_dd4);

    P24=R23*P34;
    P14=R12*R23*P34;
    
    % v1_d=transpose(R01)*(cross(wo_d,P03)+cross(wo,cross(wo,P03))+vo_d);
    v1_d=transpose(R01)*(vo_d);
    v2_d=transpose(R12)*(cross(w1_d,P14)+cross(w1,cross(w1,P14))+v1_d);
    v3_d=transpose(R23)*(cross(w2_d,P24)+cross(w2,cross(w2,P24))+v2_d);
    v4_d=transpose(R34)*(cross(w3_d,P34)+cross(w3,cross(w3,P34))+v3_d);
    

    r1=P14/2;
    vc1_d=cross(w1_d,r1)+cross(w1,cross(w1,r1))+v1_d;
    r2=P24/2;
    vc2_d=cross(w2_d,r2)+cross(w2,cross(w2,r2))+v2_d;
    r3=P34/2;
    vc3_d=cross(w3_d,r3)+cross(w3,cross(w3,r3))+v3_d;
    r4=P45/2;
    vc4_d=cross(w4_d,r4)+cross(w4,cross(w4,r4))+v4_d;

    F3=m3*vc3_d;
    N3=I3*w3_d+cross(w3,I3*w3);

    F4=m4*vc4_d;
    N4=I4*w4_d+cross(w4,I4*w4);


    %%backward
    f4=F4;
    f3=R34*F3;
    % f2=R12*f3;
    % f1=R01*f2;

    n4=N4;
    n3=N3+R34*n4+cross((P34/2),F3)+cross(P34,R34*f4);
    % n2=N2+R23*n3+cross((P23/2),F2)+cross(P23,R23*f3);
    % n1=N1+R12*n2+cross((P12/2),F1)+cross(P12,R12*f1);

    % T3=transpose(n2)*zo;
    % T2=transpose(n1)*zo;
    % T1=transpose(n0)*zo ;

    T(i)=n3(2);

    if thet2<deg2rad(-40)
        T(i)=0;
        check(i)=0;
    else
        T(i)=n3(2);
    end

    

end


t=arm.time(range)/1000;
load=arm.load(range);

Ld=load*1.4/1000;


plot(t,(Ld))
hold on
plot(t,(T))
plot(t,check)
legend('robot','model','check')

function i=I(l, m, r)
    I1=1 / 2 * (m * r^2);
    I2=1 / 12 * (m*(3 * r^2 + l^2));

    i=[I1, 0, 0;
       0, I2, 0;
       0,0,I2];
end

% function T=TI(alpha0, ai, di, thetai)
%     T=[cos(thetai), -sin(thetai)*cos(alpha0), sin(thetai)*sin(alpha0), ai*cos(thetai);
%        sin(thetai), cos(thetai)*cos(alpha0), -cos(thetai)*sin(alpha0), -ai*sin(thetai);
%        0, sin(alpha0),  cos(alpha0), di;
%        0,0,0,1];
% end

function T=TI(ai, alpha0, di, thetai)
    T=[cos(thetai), -sin(thetai), 0, ai;
       sin(thetai)*cos(alpha0), cos(thetai)*cos(alpha0), -sin(alpha0), -di*sin(thetai);
       sin(thetai)*sin(alpha0), cos(thetai)*sin(alpha0), cos(alpha0), di*cos(thetai);
       0,0,0,1];
end

% function P=PI(ai, alpha0, di, thetai)
%     P=[ai;-di*sin(thetai);di*cos(thetai)];
% end

