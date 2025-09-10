%mass
m1=0;
m2=0;
m3=.15;
m4=.17;
%length
l0=0;
l1=0;
l2=1;
le=.8;

%
r3=.2;
r4=.2;


% dhparams = [l1, 0,      0,  theta1; %01
%             0,  pi/2,   0,	theta2;%12
%             0,  theta3,	0,	0; %23
%             l2, 0,  	0,  theta4; %34
%             le, 0,      0,  0]; %4e


%Ti format: alpha0, ai, di, thetai
thett1=0;
thett2=0;
thett3=0;
thett4=0;

%matlab format: [a alpha d theta].
%DH Parameters
T01 = [l1, 0, 0, thett1];
T12 = [0, pi/2, 0, thett2];
T23 = [0, thett3, 0, 0];
T34 = [l2, 0, 0, thett4];
T4e = [le, 0, 0, 0];


%making robot
robot = rigidBodyTree('DataFormat','row');

%frame1: shoulder abduction (mass=0)
frame1 = rigidBody('body1');
frame1_jnt = rigidBodyJoint('jnt1','revolute');
setFixedTransform(frame1_jnt,T01, "dh");

frame1.Joint = frame1_jnt;
frame1.Mass = 0;
frame1.CenterOfMass = [0, 0, 0];
frame1.Inertia = [0, 0, 0, 0, 0, 0];

addBody(robot,frame1,'base')

%frame 2: shoulder flexion (mass=0)
frame2 = rigidBody('body2');
frame2_jnt = rigidBodyJoint('jnt2','revolute');
setFixedTransform(frame2_jnt,T12, "dh");

frame2.Joint = frame2_jnt;
frame2.Mass = 0;
frame2.CenterOfMass = [0, 0, 0];
frame2.Inertia = [0, 0, 0, 0, 0, 0];

addBody(robot,frame2,'body1')

%frame 3: shoulder pronation (mass=upper arm)
frame3 = rigidBody('body3');
frame3_jnt = rigidBodyJoint('jnt3','revolute');
setFixedTransform(frame3_jnt,T23, "dh");

frame3.Joint = frame3_jnt;
frame3.Mass = m3;
frame3.CenterOfMass = [l2/2, 0, 0];
frame3.Inertia = I(l2,m3,r3);

addBody(robot,frame3,'body2')

%frame 4: elbow flexion (mass=lower arm)
frame4 = rigidBody('body4');
frame4_jnt = rigidBodyJoint('jnt4','revolute');
setFixedTransform(frame4_jnt,T34, "dh");

frame4.Joint = frame4_jnt;
frame4.Mass = 1;
frame4.CenterOfMass = [le/2, 0, 0];
frame4.Inertia = I(le,m4,r4);

addBody(robot,frame4,'body3')

%frame 5:hand (mass=hand)
hand = rigidBody('body5');
hand_jnt = rigidBodyJoint('jnt5','fixed');
setFixedTransform(hand_jnt,T4e, "dh");
hand.Joint = hand_jnt;
hand.Mass = 0;

addBody(robot,hand,'body4')


%show robot details
showdetails(robot)

arm=readtable('larm_5.csv');
pwm=arm.position;
angle=((pwm-mean(pwm(1:10)))/4095)*360;

theta2=transpose(deg2rad(angle));
n=length(theta2);

theta1=zeros(1,n);
theta3=zeros(1,n);
theta4=deg2rad(ones(1,n)*45);

dt=0.02;

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


tt=zeros(1,n);

%orientation
for i=1:n
    q = [theta1(i),theta2(i),theta3(i),theta4(i)];
    vel=[theta_d1(i),theta_d2(i),theta_d3(i),theta_d4(i)];
    acc=[theta_dd1(i),theta_dd2(i),theta_dd3(i),theta_dd4(i)];
    
    robot.Gravity=[0 0 -9.80665];
    
    T=inverseDynamics(robot,q,vel,acc);
    tt(i)=T(3);
end

plot(normalize(tt))
hold on
plot(normalize(arm.load))


function i=I(l, m, r)
    I1=1 / 2 * (m * r^2);
    I2=1 / 12 * (m*(3 * r^2 + l));

    i=[I1, I2, I2,0,0,0];
end


