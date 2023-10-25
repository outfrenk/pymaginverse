#!/bin/csh -f
\rm spec.eps

set A = -3005
set start=-8000
set endt=1990
set MOD1 = new1-131-2.it20
set MOD2 = new1-151-2.it20

~/archfield/progdi/Avcoef10<<EOF
coeffile   =$MOD1
outfile    =avcoefs.dat
outfilesv  =sv.dat
$start   starttime 
$endt   endtime 
4      damping
0.547  depth  
7      tdamping
1      tdampflag
EOF
mv spec.dat specav1.dat

~/archfield/progdi/Avcoef10<<EOF
coeffile   =$MOD2
outfile    =avcoefs.dat
outfilesv  =sv.dat
$start starttime 
$endt endtime 
4      damping
0.547  depth  
7      tdamping
1      tdampflag
EOF
mv spec.dat specav2.dat

../Spec<<EOF
$A
$MOD1
EOF
mv spec.dat spec1.dat

../Spec<<EOF
$A
$MOD2
EOF
mv spec.dat spec2.dat

../Exspec<<EOF
$A
CALS10k.2
0
EOF
mv spec.dat spec3.dat

plotxy<<EOF
title \sim\
weight 12
character 0.12
color black
output spec.ps
frame +box
logxy linlog
xlimit 2.5 0 11 2
xlabel SH degree
symbol 19 0.15

ylimit 2.5 5e6 1e11
ylabel Power [nT\sup{2}]
note (0.2 2.3 in) MF CMB avg.
note (-0.9 -0.5 in) a) 
mode 20 1 3
file TOYavg.spec
color black
read
file specav1.dat
color blue
read
file specav2.dat
color red
read
color black
plot 1 6.5

ylimit 2.5 1e2 1e6
note
note (0.2 2.3 in) SV CMB avg.
ylabel Power [nT\sup{2}/yr\sup{2}]
note (-0.9 -0.5 in) b) 
mode 20 1 5
file TOYavg.spec
color black
read
file specav1.dat
color blue
read
file specav2.dat
color red
read
color black
plot 3.5 0

ylimit 2.5 5e6 1e11
ylabel Power [nT\sup{2}]
note
note (0.2 2.3 in) MF CMB $A
note (-0.9 -0.5 in) c) 
mode 20 1 3
file spec3.dat
color black
read
file spec1.dat
color blue
read
file spec2.dat
color red
read
color black
plot -3.5 -3.5

ylimit 2.5 1e2 1e6
note
note (0.2 2.3 in) SV CMB $A
ylabel Power [nT\sup{2}/yr\sup{2}]
note (-0.9 -0.5 in) d) 
mode 20 1 5
file spec3.dat
color black
read
file spec1.dat
color blue
read
file spec2.dat
color red
read
color black
plot 3.5 0
stop
EOF
ps2eps spec.ps
gv spec.eps &
