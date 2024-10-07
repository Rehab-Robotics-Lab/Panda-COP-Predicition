function T=Ti(alpha0, ai, di, thetai)
    T=[cos(thetai), -sin(thetai), 0, ai;
       sin(thetai)*cos(alpha0), cos(thetai)*cos(alpha0), -sin(alpha0), -sin(alpha0)*di;
       sin(thetai)*sin(alpha0), cos(thetai)*sin(alpha0),  cos(alpha0), cos(alpha0)*di;
       0,0,0,1];
end