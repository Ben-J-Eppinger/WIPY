 set term wxt
 #set term gif
 #set output "points_per_wavelength_histogram_P_in_fluid.gif"

 set boxwidth    6.00000005E-03
 set xlabel "Range of min number of points per P wavelength in fluid"
 set ylabel "Percentage of elements (%)"
 set loadpath "./OUTPUT_FILES/"
 plot "points_per_wavelength_histogram_P_in_fluid.txt" with boxes
 pause -1 "hit any key..."
