set term postscript eps solid enhanced color 52

set size 2.2,2.2
set bar 1.000000 front
set border 3 front lt black linewidth 1.000 dashtype solid
set boxwidth 0.75 absolute
set style fill solid 1.00 border lt -1
set style circle radius graph 0.02, first 0.00000, 0.00000 
set style ellipse size graph 0.05, 0.03, first 0.00000 angle 0 units xy
set grid nopolar
set grid noxtics nomxtics ytics nomytics noztics nomztics \
 nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics
set grid layerdefault lt 0 linewidth 0.500,  lt 0 linewidth 0.500
#set key outside right top vertical Left reverse noenhanced autotitle columnhead nobox
set key outside horiz center top reverse noenhanced autotitle columnhead nobox
set key invert samplen 4 spacing 1 width 0 height 0 
set style histogram rowstacked title textcolor lt -1
set style textbox transparent margins  1.0,  1.0 border
set style data histograms
set xtics border in scale 0,0 nomirror rotate by -270  autojustify
unset ytics
set title ""
set ylabel "Frequency of observation" 
set yrange [ 0.00000 : 100.000 ] noreverse nowriteback
set colorbox vertical origin screen 0.9, 0.2, 0 size screen 0.05, 0.6, 0 front bdefault

set output 'fobs.eps'
plot 'sign_domain' using (100.*$2/$6):xtic(1) t column(2), for [i=3:5] '' using (100.*column(i)/column(6)) title column(i)

reset
set term postscript eps solid enhanced color 52
set size 1.3,1.3
unset key
#set key inside right top vertical Left reverse enhanced autotitle nobox
#set grid y
set xtics 8
#unset ytics
set xlabel ""
set ylabel ""
set ytics 0.002
set title "Cardiac (speed)" 
set output 'dynamics_speed_icu1.eps'
plot [][] "dynamics0_icu1" u 2:5 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu1" u 2:5 with lines lw 15 lt 7 t "Death" smooth acsplines
set ylabel "" 
set title "Cardiac (acceleration)"
set output 'dynamics_acc_icu1.eps'
plot [][] "dynamics0_icu1" u 2:6 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu1" u 2:6 with lines lw 15 lt 7 t "Death" smooth acsplines
unset key
#set key inside left top vertical Left reverse enhanced autotitle nobox
set ylabel "" 
set ytics 0.02
set title "Cardiac (distance)"
set output 'dynamics_dist_icu1.eps'
plot [][] "dynamics0_icu1" u 2:10 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu1" u 2:10 with lines lw 15 lt 7 t "Death" smooth acsplines
reset
set term postscript eps solid enhanced color 52
set size 1.3,1.3
unset key
#set key inside right top vertical Left reverse enhanced autotitle nobox
#set grid y
set xtics 8
#unset ytics
set xlabel "" 
set ylabel "" 
set title "Coronary (speed)" 
set ytics 0.0004
set output 'dynamics_speed_icu2.eps'
plot [][] "dynamics0_icu2" u 2:5 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu2" u 2:5 with lines lw 15 lt 7 t "Death" smooth acsplines
set ylabel ""
set title "Coronary (acceleration)"
set output 'dynamics_acc_icu2.eps'
plot [][] "dynamics0_icu2" u 2:6 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu2" u 2:6 with lines lw 15 lt 7 t "Death" smooth acsplines
unset key
#set key inside left top vertical Left reverse enhanced autotitle nobox
set ylabel ""
set ytics 0.02
set title "Coronary (distance)" 
set output 'dynamics_dist_icu2.eps'
plot [][] "dynamics0_icu2" u 2:10 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu2" u 2:10 with lines lw 15 lt 7 t "Death" smooth acsplines
reset
set term postscript eps solid enhanced color 52
set size 1.3,1.3
unset key
#set key inside right top vertical Left reverse enhanced autotitle nobox
#set grid y
set xtics 8 
#unset ytics
set xlabel ""
set ylabel ""
set ytics 0.001
set title "Medical (speed)" 
set output 'dynamics_speed_icu3.eps'
plot [][] "dynamics0_icu3" u 2:5 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu3" u 2:5 with lines lw 15 lt 7 t "Death" smooth acsplines
set ylabel ""
set title "Medical (acceleration)"
set output 'dynamics_acc_icu3.eps'
plot [][] "dynamics0_icu3" u 2:6 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu3" u 2:6 with lines lw 15 lt 7 t "Death" smooth acsplines
unset key
#set key inside left top vertical Left reverse enhanced autotitle nobox
set ylabel ""
set ytics 0.02
set title "Medical (distance)" 
set output 'dynamics_dist_icu3.eps'
plot [][] "dynamics0_icu3" u 2:10 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu3" u 2:10 with lines lw 15 lt 7 t "Death" smooth acsplines
reset
set term postscript eps solid enhanced color 52
set size 1.3,1.3
unset key
#set key inside right top vertical Left reverse enhanced autotitle nobox
#set grid y
set xtics 8
#unset ytics
set xlabel ""
set ylabel ""
set ytics 0.01
set title "Surgical (speed)"
set output 'dynamics_speed_icu4.eps'
plot [][] "dynamics0_icu4" u 2:5 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu4" u 2:5 with lines lw 15 lt 7 t "Death" smooth acsplines
set ylabel ""
set title "Surgical (acceleration)"
set output 'dynamics_acc_icu4.eps'
plot [][] "dynamics0_icu4" u 2:6 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu4" u 2:6 with lines lw 15 lt 7 t "Death" smooth acsplines
unset key
#set key inside left top vertical Left reverse enhanced autotitle nobox
set ylabel ""
set ytics 0.01
set title "Surgical (distance)"
set output 'dynamics_dist_icu4.eps'
plot [][] "dynamics0_icu4" u 2:10 with lines lw 15 lt 3 t "Survival" smooth acsplines, "dynamics1_icu4" u 2:10 with lines lw 15 lt 7 t "Death" smooth acsplines



reset
set term postscript eps solid enhanced color 47
unset key
#set key inside right top vertical Left reverse enhanced autotitle nobox
#set grid y
set xtics 8
unset ytics
set xlabel "" 
set ylabel "speed" 
set title "Cardiac (raw)" 
set output 'raw_speed_icu1.eps'
plot [][] "raw0_icu1" u 2:5 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu1" u 2:5 with lines lw 15 lt 7 t "Death" smooth csplines
set ylabel "acceleration" 
set output 'raw_acc_icu1.eps'
plot [][] "raw0_icu1" u 2:6 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu1" u 2:6 with lines lw 15 lt 7 t "Death" smooth csplines
unset key
#set key inside left top vertical Left reverse enhanced autotitle nobox
set ylabel "distance" 
set output 'raw_dist_icu1.eps'
plot [][] "raw0_icu1" u 2:10 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu1" u 2:10 with lines lw 15 lt 7 t "Death" smooth csplines
reset
set term postscript eps solid enhanced color 47
unset key
#set key inside right top vertical Left reverse enhanced autotitle nobox
#set grid y
set xtics 8
unset ytics
set xlabel "" 
set ylabel "speed" 
set title "Coronary (raw)" 
set output 'raw_speed_icu2.eps'
plot [][] "raw0_icu2" u 2:5 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu2" u 2:5 with lines lw 15 lt 7 t "Death" smooth csplines
set ylabel "acceleration"
set output 'raw_acc_icu2.eps'
plot [][] "raw0_icu2" u 2:6 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu2" u 2:6 with lines lw 15 lt 7 t "Death" smooth csplines
unset key
#set key inside left top vertical Left reverse enhanced autotitle nobox
set ylabel "distance"
set output 'raw_dist_icu2.eps'
plot [][] "raw0_icu2" u 2:10 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu2" u 2:10 with lines lw 15 lt 7 t "Death" smooth csplines
reset
set term postscript eps solid enhanced color 47
unset key
#set key inside right top vertical Left reverse enhanced autotitle nobox
#set grid y
set xtics 8 
unset ytics
set xlabel "" 
set ylabel "speed" 
set title "Medical (raw)" 
set output 'raw_speed_icu3.eps'
plot [][] "raw0_icu3" u 2:5 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu3" u 2:5 with lines lw 15 lt 7 t "Death" smooth csplines
set ylabel "acceleration"
set output 'raw_acc_icu3.eps'
plot [][] "raw0_icu3" u 2:6 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu3" u 2:6 with lines lw 15 lt 7 t "Death" smooth csplines
unset key
#set key inside left top vertical Left reverse enhanced autotitle nobox
set ylabel "distance"
set output 'raw_dist_icu3.eps'
plot [][] "raw0_icu3" u 2:10 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu3" u 2:10 with lines lw 15 lt 7 t "Death" smooth csplines
reset
set term postscript eps solid enhanced color 47
unset key
#set key inside right top vertical Left reverse enhanced autotitle nobox
#set grid y
set xtics 8
unset ytics
set xlabel ""
set ylabel "speed"
set title "Surgical (raw)"
set output 'raw_speed_icu4.eps'
plot [][] "raw0_icu4" u 2:5 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu4" u 2:5 with lines lw 15 lt 7 t "Death" smooth csplines
set ylabel "acceleration"
set output 'raw_acc_icu4.eps'
plot [][] "raw0_icu4" u 2:6 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu4" u 2:6 with lines lw 15 lt 7 t "Death" smooth csplines
unset key
#set key inside left top vertical Left reverse enhanced autotitle nobox
set ylabel "distance" 
set output 'raw_dist_icu4.eps'
plot [][] "raw0_icu4" u 2:10 with lines lw 15 lt 3 t "Survival" smooth csplines, "raw1_icu4" u 2:10 with lines lw 15 lt 7 t "Death" smooth csplines
