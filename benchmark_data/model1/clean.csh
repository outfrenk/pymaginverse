#!/bin/csh -f
\rm -f allnorms
touch allnorms
echo 'resid        rms   s-norm   lambda   t-norm   tau   dipfac   gufmfac   rmsI   rmsD   rmsF' >! allnorms

cp modintd.dipole modintd

set a = 1
while ($a < 11)
    ~/archfield/progdi/Oneclean < inputtd
    ~/archfield/progdi/Misfitclean < inputtd
    cat normres >> allnorms
    \rm modintd
    cp newmod modintd
    cp newmod newmod.it$a
@ aa = $a
@ a = $aa + 1
end

