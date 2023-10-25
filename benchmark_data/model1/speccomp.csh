#!/bin/csh -f
\rm spec.dat
\rm spec1.dat
\rm avcoefs.dat
\rm sv.dat
\rm speccomp.ps

set A='new5-131-1.it30'
set B='new5-131-1.it10'
set start=-8000
set endt=1990

~/reverse/code/Avcoef<<EOF
coeffile   =$A
outfile    =avcoefs.dat
outfilesv  =sv.dat
$start   starttime 
$endt   endtime 
4      damping
0.547  depth  
7      tdamping
1      tdampflag
EOF
mv spec.dat spec1.dat

~/reverse/code/Avcoef<<EOF
coeffile   =$B
outfile    =avcoefs.dat
outfilesv  =sv.dat
$start starttime 
$endt endtime 
4      damping
0.547  depth  
7      tdamping
1      tdampflag
EOF

plotxy<<EOF
color red
note (0.5 2.7 in) $A
color blue
note (0.5 2.9 in) $B
color black
note (3.5 2.9 in) Time interval:
note (3.5 2.7 in) $start  to  $endt
output speccomp.ps
xlimit 2.5
ylimit 2.5
frame +box
character 0.1
xlabel SH degree
ylabel power [nT\sup{2}]
weight 12
  weight 
logxy linlog
note (0.2 2.3 in) Earth's surface
note (-0.8 -0.4 in) a)  
file spec.dat
mode 20 1 2
symbol 19 0.15
color blue
read
file spec1.dat
color red
read
file /home/monika/archfield3/new10k/spec.CALS10k.2
symbol 20 0.15
color green
read
file /home/monika/magmodel/gufm/gufmspec.dat
symbol 21 0.15
color black
read
color black
plot 1 6.5
note
note (0.2 2.3 in) CMB
note (-0.8 -0.4 in) b) 
file spec.dat
mode 20 1 3
symbol 19 0.15
color blue
read
file spec1.dat
color red
read
file /home/monika/archfield3/new10k/spec.CALS10k.2
symbol 20 0.15
color green
read
file /home/monika/magmodel/gufm/gufmspec.dat 
symbol 21 0.15
color black
read
color black
plot 3.5 0
ylabel SV power [nT\sup{2}/yr\sup{2}]
note
note (-0.8 2.3 in) c) 
file spec.dat
mode 20 1 4
color blue
symbol 19 0.15
read
file spec1.dat
color red
read
file /home/monika/archfield3/new10k/spec.CALS10k.2
symbol 20 0.15
color green
read
file /home/monika/magmodel/gufm/gufmspec.dat
symbol 21 0.15
color black
read
color black
plot -3.5 -3.5
note
note (-0.8 2.3 in) d) 
file spec.dat
mode 20 1 5
symbol 19 0.15
color blue
read
file spec1.dat
color red
read
file /home/monika/archfield3/new10k/spec.CALS10k.2
symbol 20 0.15
color green
read
file /home/monika/magmodel/gufm/gufmspec.dat
symbol 21 0.15
color black 
read
color black
plot 3.5 0
stop
EOF
gv speccomp.ps &
