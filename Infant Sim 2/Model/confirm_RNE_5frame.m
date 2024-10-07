%%parameters
%mass
m1=0;
m2=0;
m3=.377;
m4=.3;
%length
l0=0;
l1=0;
l2=.1;
le=.08;
%
r3=.02;
r4=.015;
%intertia
I1=eye(3);
I2=eye(3);
I3=I(l2,m3,r3);
I4=I(le,m4,r4);

arm=readtable(['larm_5' ...
    '.csv']);

[a,b]=size(arm);

range=1:a;

% arm_ang=medfilt1(arm.arm1Angle,5);

freq_cuttoff=2;
n_order=2;
rate=60;
[b,a]=butter(n_order,freq_cuttoff/(rate/2));

arm_ang=(((arm.position-1835)/4095)*360)-45;
% arm_ang = filtfilt(b,a,arm_ang);

theta2=transpose(deg2rad(arm_ang(range)));
theta1=zeros(length(theta2));
theta3=zeros(length(theta2));
t4=45;
theta4=deg2rad(ones(length(theta2))*t4);



% %thetas
% theta1=deg2rad([0,0,0,0,0,0,0]);
% theta2=deg2rad([50,40,20,30,40,50,60]);
% theta3=deg2rad([0,0,0,0,0,0,0]);
% t4=30;
% theta4=deg2rad([1,1,1,1,1,1,1]*t4);

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
vo_d=[0;-9.81;0];
wo_d=[0;0;0];

zo=[0;0;1];

fe=[0;0;0];
ne=[0;0;0];

T=zeros('like',theta2);
check=zeros('like',theta2);

for i=1:length(theta2)
% for i=97
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

    %T,R and P  
    T01 = Ti(0, 0, 0, thet1);
    T01 = TI(pi/2, 0 , 0, pi/2+thet1);
    P01=T01(1:3,4);
    R01=T01(1:3,1:3);
    
    T12 = Ti(pi/2, 0, 0, thet2);
    T12 = TI(pi/2, 0 , 0, thet2-pi/2);
    P12=T12(1:3,4);
    R12=T12(1:3,1:3);
    
    T23 = Ti(thet3, l2, 0, 0);
    T23=TI(-pi/2, 0 , l2, thet3);
    P23=T23(1:3,4);
    R23=T23(1:3,1:3);
    
    T34 = Ti(0, le, 0, thet4);
    T41=TI(pi/2, 0 , 0, thet4+pi/2);
    T42=TI(0, 0 , le, 0);

    T34=T41*T42;
    P34=T34(1:3,4);
    R34=T34(1:3,1:3);
    
    T4e = Ti(0, le, 0, 0);
    P4e=T4e(1:3,4);
    R4e=T4e(1:3,1:3);

    T13=T01*T12*T23;
    P13=T13(1:3,4);
    R13=T13(1:3,1:3);

    % thet13=[thet1;]

    %%forward
    %FRAME1
    w1=transpose(R01)*wo+zo*thet_d1;
    w1_d=(transpose(R01)*wo_d)+cross(transpose(R01)*wo,zo*thet_d1)+(zo*thet_dd1);
    v1_d=transpose(R01)*(cross(wo_d,P01)+cross(wo,cross(wo,P01))+vo_d);
    vc1_d=cross(w1_d,P12/2)+cross(w1,cross(w1,P12/2))+v1_d;
    F1=m1*vc1_d;
    N1=I1*w1_d+cross(w1,I1*w1);

    %FRAME2
    w2=transpose(R12)*w1+zo*thet_d2;
    w2_d=(transpose(R12)*w1_d)+cross(transpose(R12)*w1,zo*thet_d2)+(zo*thet_dd2);
    check(i)=w2_d(3);
    v2_d=transpose(R12)*(cross(w1_d,P12)+cross(w1,cross(w1,P12))+v1_d);
    vc2_d=cross(w2_d,P23/2)+cross(w2,cross(w2,P23/2))+v2_d;
    F2=m2*vc2_d;
    N2=I2*w2_d+cross(w2,I2*w2);

    %FRAME3
    w3=transpose(R23)*w2+zo*thet_d3;
    w3_d=(transpose(R23)*w2_d)+cross(transpose(R23)*w2,zo*thet_d3)+(zo*thet_dd3);
    v3_d=transpose(R23)*(cross(w2_d,P23)+cross(w2,cross(w2,P23))+v2_d);
    vc3_d=cross(w3_d,P34/2)+cross(w3,cross(w3,P34/2))+v3_d;
    F3=m3*vc3_d;
    N3=I3*w3_d+cross(w3,I3*w3);

    %FRAME4
    w4=transpose(R34)*w3+zo*thet_d4;
    w4_d=(transpose(R34)*w3_d)+cross(transpose(R34)*w3,zo*thet_d4)+(zo*thet_dd4);
    v4_d=transpose(R34)*(cross(w3_d,P34)+cross(w3,cross(w3,P34))+v3_d);
    vc4_d=cross(w4_d,P4e/2)+cross(w4,cross(w4,P4e/2))+v4_d;
    F4=m4*vc4_d;
    N4=I4*w4_d+cross(w4,I4*w4);
    
    F=[F1,F2,F3,F4];
    N=[N1,N2,N3,N4];
    
    %%backward
    f4=R4e*fe+F4;
    f3=R34*f4+F3;
    f2=R23*f3+F2;
    f1=R12*f2+F1;

    n4=N4+R4e*ne+cross((P4e/2),F4)+cross(P4e,R4e*fe);
    n3=N3+R34*n4+cross((P34/2),F3)+cross(P34,R34*f4);
    n2=N2+R23*n3+cross((P23/2),F2)+cross(P23,R23*f3);
    n1=N1+R12*n2+cross((P12/2),F1)+cross(P12,R12*f1);

    T4=transpose(n4)*zo;
    T3=transpose(n3)*zo;
    T2=transpose(n2)*zo;
    T1=transpose(n1)*zo;

    T(i)=T2;
   



end

t=arm.time(range)/1000;
% load=arm.arm1Load(range);
load=arm.load(range);

Tn=normalize(T);

dt=diff(t);

Ld=load*1.4/1000;

% plot(t,check)
% [min(check),max(check)]

figure
plot(t,Ld)
hold on
plot(t,T)
ylabel('N.m')
legend('robot','model')

figure 
plot(t, rad2deg(theta2))
ylabel('Deg')

% legend('robot','angle','model')


function i=I(l, m, r)
    I1=1 / 2 * (m * r^2);
    I2=1 / 12 * (m*(3 * r^2 + l^2));

    i=[I1, 0, 0;
       0, I2, 0;
       0,0,I2];
end

function T=TI(alpha0, ai, di, thetai)
    T=[cos(thetai), -sin(thetai)*cos(alpha0), sin(thetai)*sin(alpha0), ai*cos(thetai);
       sin(thetai), cos(thetai)*cos(alpha0), -cos(thetai)*sin(alpha0), -ai*sin(thetai);
       0, sin(alpha0),  cos(alpha0), di;
       0,0,0,1];
end

