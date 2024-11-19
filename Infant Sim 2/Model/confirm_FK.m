theta1=0;
theta2=0;
theta3=-60;
theta4=-80;

l_up=.2;
l_low=.24;


robot = rigidBodyTree;

body1 = rigidBody('body1');
jnt1 = rigidBodyJoint('jnt1','revolute');
T01=TI(0 , pi/2, 0, deg2rad(theta1)+pi/2);
[R1,P1]=RP(T01);
setFixedTransform(jnt1,T01);
body1.Joint = jnt1;
addBody(robot,body1,'base')

body2 = rigidBody('body2');
jnt2 = rigidBodyJoint('jnt2','revolute');
T12=TI(0, pi/2 , 0, deg2rad(theta2)+(3*pi/2));
[R2,P2]=RP(T12);
setFixedTransform(jnt2,T12);
body2.Joint = jnt2;
addBody(robot,body2,'body1')

body3 = rigidBody('body3');
jnt3 = rigidBodyJoint('jnt3','revolute');
T23=TI(0, -pi/2, 0, pi+deg2rad(theta3));
[R3,P3]=RP(T23);
setFixedTransform(jnt3,T23);
body3.Joint = jnt3;
addBody(robot,body3,'body2')

body4 = rigidBody('body4');
jnt4 = rigidBodyJoint('jnt4','revolute');
T34=TI(l_up, pi/2 , 0, deg2rad(theta4));
[R4,P4]=RP(T34);
setFixedTransform(jnt4,T34);
body4.Joint = jnt4;
addBody(robot,body4,'body3')

body5 = rigidBody('body5');
jnt5 = rigidBodyJoint('jnt5','revolute');
T45=TI(l_low, 0 , 0, 0);
[R5,P5]=RP(T45);
setFixedTransform(jnt5,T45);
body5.Joint = jnt5;
addBody(robot,body5,'body4')


q = homeConfiguration(robot);
t_01=getTransform(robot,q, 'body4','body1');
p_01=t_01(1:3,4);
t_02=getTransform(robot,q, 'body4','body2');
p_02=t_02(1:3,4);
t_03=getTransform(robot,q, 'body4','body3');
p_03=t_03(1:3,4);


show(robot);
% gui = interactiveRigidBodyTree(robot,"MarkerScaleFactor",0.25);



% Ti(alpha0, ai, di, thetai)
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

function [R,P]=RP(Ti)
    P=Ti(1:3,4);
    R=Ti(1:3,1:3);
end