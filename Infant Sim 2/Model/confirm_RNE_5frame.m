

l0=0;
l1=0;
l2=1;
le=.8;

theta1=0;
theta2=50;
theta3=0;
theta4=30;

% dhparams = [l1, 0,      0,  theta1; %01
%             0,  pi/2,   0,	theta2;%12
%             0,  theta3,	0,	0; %23
%             l2, 0,  	0,  theta4; %34
%             le, 0,      0,  0]; %4e


%Ti format: alpha0, ai, di, thetai


%matlab format: [a alpha d theta].
%DH Parameters
T01 = [l1, 0, 0, theta1];
T12 = [0, pi/2, 0, theta2];
T23 = [0, theta3, 0, 0];
T34 = [l2, 0, 0, theta4];
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
frame3.Mass = 1;
frame3.CenterOfMass = [l2/2, 0, 0];
frame3.Inertia = [0.02, 0.0933, 0.0933, 0, 0, 0];

addBody(robot,frame3,'body2')

%frame 4: elbow flexion (mass=lower arm)
frame4 = rigidBody('body4');
frame4_jnt = rigidBodyJoint('jnt4','revolute');
setFixedTransform(frame4_jnt,T34, "dh");

frame4.Joint = frame4_jnt;
frame4.Mass = 1;
frame4.CenterOfMass = [le/2, 0, 0];
frame4.Inertia = [0.02, 0.0767, 0.0767, 0, 0, 0];

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

%orientation
q = [deg2rad(theta1),deg2rad(theta2),deg2rad(theta3),deg2rad(theta4)];
vel=[0,0,0,0];
acc=[0,0,0,0];

robot.Gravity=[0 0 -9.80665];

inverseDynamics(robot,q,vel,acc)

show(robot);



