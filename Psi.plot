set terminal pngcairo size 1440,1080
set output 'PsiMid.png'
set title 'Axial magnetic flux at midplane'; set xlabel 'R(m)'; set ylabel 'Psi(Wb)'
 plot [0:0.4] [0:] 'Psi.dat' using 1:2 w lines title 'Vacuum Field', 'Psi.dat' using 1:3 w lines title 'Plasma'

