set terminal pngcairo size 1440,1080
set output 'Bz.png'
set title 'Axial magnetic field at midplane'; set xlabel 'R(m)'; set ylabel 'B_z'
 plot [0:0.4] [0:] 'Psi.dat' using 1:4 w lines title 'Vacuum Field', 'Psi.dat' using 1:5 w lines title 'Plasma'

