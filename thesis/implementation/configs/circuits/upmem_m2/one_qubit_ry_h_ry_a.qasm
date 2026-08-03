OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
ry(pi/5) q[0];
h q[0];
ry(pi/9) q[0];
