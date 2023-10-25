#!/bin/csh -f
\rm -f allnorms
touch allnorms
echo 'resid        rms   s-norm   lambda   t-norm   tau   dipfac   gufmfac   rmsI   rmsD   rmsF' >! allnorms
#spatial
   foreach j (-13)
   foreach i (5)
#temporal
    foreach l (-1)
    foreach k (1)
#axial dipole constraint
    foreach m (0)
    foreach n (0)
#fit to gufm at end
    foreach o (0)
    foreach p (0)
cp modintd.dipole modintd
\rm -f dall$i$j$k$l
touch dall$i$j$k$l
echo 'resid        rms   s-norm   lambda   t-norm   tau   dipfac   gufmfac   rmsI   rmsD   rmsF' >! allnorms
set a = 1
while ($a < 11)
    head -3 inputtd > InPtEmP
    echo "New model  =newmod" >> InPtEmP
    head -8 inputtd | tail -4 >> InPtEmP 
    echo $i".0d"$j"   Damping    ="  >> InPtEmP
    head -12 inputtd | tail -3 >> InPtEmP
    echo $m".0d"$n"  "$o".0d"$p"    dipolecoeff and gufm">>InPtEmP
    echo $k".0d"$l"   add. tdamp =" >> InPtEmP
    tail -n +15 inputtd >> InPtEmP
    ~/archfield/progdi/Onestep < InPtEmP
    ~/archfield/progdi/Misfit < InPtEmP
    cat normres >> dall$i$j$k$l
    \rm modintd
    cp newmod modintd
    cp newmod dnew$i$j$k$l.it$a
@ aa = $a
@ a = $aa + 1
end
    end
    end
    end
    end
    end
    end
    end
end
