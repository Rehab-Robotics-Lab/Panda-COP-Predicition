%%%
syms t1 t2 t3 t4
r1=rotz(pi)*rotz(t1)
r2=rotx(t2)
r3=roty(t3)
r4=rotx(t4)

R=r1*r2*r3*r4

%%  FLSDNS

syms c1 s1 c2 s2 c3 s3 c4 s4 X Y Z

% eqn=[s4*(c1*s3 + c3*s1*s2) - c2*c4*s1;s4*(s1*s3 - c1*c3*s2) + c1*c2*c4;c4*s2 + c2*c3*s4]==[X,Y,Z]
ex=-s4*(c1*s3 + c3*s1*s2) + c2*c4*s1;
ey=-s4*(s1*s3 +c1*c3*s2) - c1*c2*c4;
ez=c4*s2 + c2*c3*s4;
eqn1=[ey ;ez ]==[Y;Z]
eqn2=[ ex; ez]==[X;Z]
eqn3=[ex ; ey]==[X;Y]

solve(eqn1,[c3,s3])
solve(eqn2,[c3,s3])
solve(eqn3,[c3,s3])
