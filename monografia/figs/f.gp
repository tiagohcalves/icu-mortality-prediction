set term postscript eps solid enhanced color 40

set size 1,1.2

set view map

set xlabel ""
set ylabel ""
set cblabel ""

set tic scale 0

set palette rgbformulae 22,13,10
#set palette negative
set palette positive 

set cbrange [-1.0:1.0]

unset key
unset xtics
unset ytics
unset cbtics
set xrange [0:39]
set yrange [0:39]

set title "Cardiac (CNN-LSTM)"
set output "cardiac.eps"
splot 'cardiac' matrix with image

set title "Coronary (CNN-LSTM)"
set output "coronary.eps"
splot 'coronary' matrix with image

set title "Medical (CNN-LSTM)"
set output "medical.eps"
splot 'medical' matrix with image

set title "Surgical (CNN-LSTM)"
set output "surgical.eps"
splot 'surgical' matrix with image

set title "Cardiac (raw)"
set output "raw_cardiac.eps"
splot 'raw_cardiac' matrix with image

set title "Coronary (raw)"
set output "raw_coronary.eps"
splot 'raw_coronary' matrix with image

set title "Medical (raw)"
set output "raw_medical.eps"
splot 'raw_medical' matrix with image

set title "Surgical (raw)"
set output "raw_surgical.eps"
splot 'raw_surgical' matrix with image

reset
set term postscript eps solid enhanced color 28#35

set grid ytics lc rgb "#bbbbbb" lw 1 lt 0
set grid xtics lc rgb "#bbbbbb" lw 1 lt 0

set key bottom right
set ylabel "AUC"
set output "auc_time.eps"
plot [5:48][0.68:] "auc_time" u 1:3 t "Cardiac" w lines lw 8 lt 1 smooth bezier, "auc_time" u 1:2 t "Coronary" w lines lw 8 lt 2 smooth bezier, "auc_time" u 1:4 t "Medical" w lines lw 8 smooth bezier, "auc_time" u 1:5 t "Surgical" w lines lw 8 smooth bezier#, "auc_time" u 1:6 t "Overall" w lines lw 8 lt 7 smooth bezier

set output "auc_time2.eps"
plot [5:48][0.68:] "auc_time" u 1:3 t "Cardiac" w lines lw 8 lt 1 smooth bezier, "auc_time" u 1:2 t "Coronary" w lines lw 8 lt 2 smooth bezier, "auc_time" u 1:4 t "Medical" w lines lw 8 lt 3 smooth bezier, "auc_time" u 1:5 t "Surgical" w lines lw 8 lt 4 smooth bezier, "auc_time2" u 1:3 t "" w line dt '.' lw 8 lt 1 smooth bezier, "auc_time2" u 1:2 t "" w line dt '.' lw 8 lt 2 smooth bezier, "auc_time2" u 1:4 t "" w line dt '.' lw 8 lt 3 smooth bezier, "auc_time2" u 1:5 t "" w line dt '.' lw 8 lt 4 smooth bezier


reset
set term postscript eps solid enhanced color 32#35

set grid ytics lc rgb "#bbbbbb" lw 1 lt 0
set grid xtics lc rgb "#bbbbbb" lw 1 lt 0

set ytics 0.02

set ylabel "AUC"

set key bottom right

set output "auc_time3.eps"
#plot [5:48][0.66:] "auc_time" u 1:3 t "Cardiac" w lines lw 8 lt 1 smooth csplines, "auc_time" u 1:2 t "Coronary" w lines lw 8 lt 2 smooth csplines, "auc_time" u 1:4 t "Medical" w lines lw 8 lt 3 smooth csplines, "auc_time" u 1:5 t "Surgical" w lines lw 8 lt 4 smooth csplines, "auc_time4" u 1:3 t "" w line dt '.' lw 8 lt 1 smooth csplines, "auc_time4" u 1:2 t "" w line dt '.' lw 8 lt 2 smooth csplines, "auc_time4" u 1:4 t "" w line dt '.' lw 8 lt 3 smooth csplines, "auc_time4" u 1:5 t "" w line dt '.' lw 8 lt 4 smooth csplines
plot [5:48][0.66:] "auc_time" u 1:3 t "Cardiac" w lines lw 8 lt 1 smooth acsplines, "auc_time" u 1:2 t "Coronary" w lines lw 8 lt 2 smooth acsplines, "auc_time" u 1:4 t "Medical" w lines lw 8 lt 3 smooth acsplines, "auc_time" u 1:5 t "Surgical" w lines lw 8 lt 4 smooth acsplines#, "auc_time4" u 1:3 t "" w line dt '.' lw 8 lt 1 smooth acsplines, "auc_time4" u 1:2 t "" w line dt '.' lw 8 lt 2 smooth acsplines, "auc_time4" u 1:4 t "" w line dt '.' lw 8 lt 3 smooth acsplines, "auc_time4" u 1:5 t "" w line dt '.' lw 8 lt 4 smooth acsplines
#plot [5:48][0.66:] "auc_time" u 1:3 t "Cardiac" w lines lw 8 lt 1 smooth sbezier, "auc_time" u 1:2 t "Coronary" w lines lw 8 lt 2 smooth sbezier, "auc_time" u 1:4 t "Medical" w lines lw 8 lt 3 smooth sbezier, "auc_time" u 1:5 t "Surgical" w lines lw 8 lt 4 smooth sbezier, "auc_time4" u 1:3 t "" w line dt '.' lw 8 lt 1 smooth sbezier, "auc_time4" u 1:2 t "" w line dt '.' lw 8 lt 2 smooth sbezier, "auc_time4" u 1:4 t "" w line dt '.' lw 8 lt 3 smooth sbezier, "auc_time4" u 1:5 t "" w line dt '.' lw 8 lt 4 smooth sbezier

set ytics 1
set key top
set ylabel "Gains (%)"
set output "auc_time4.eps"
#plot [5:48][:] "auc_time5" u 1:(100*$3) t "Cardiac" w lines lw 8 lt 1 smooth acsplines, "auc_time5" u 1:(100*$2) t "Coronary" w lines lw 8 lt 2 smooth acsplines, "auc_time5" u 1:(100*$4) t "Medical" w lines lw 8 smooth acsplines, "auc_time5" u 1:(100*$5) t "Surgical" w lines lw 8 smooth acsplines
plot [5:48][:] "auc_time6" u 1:(100*$3) t "Cardiac" w lines lw 8 lt 1 smooth acsplines, "auc_time6" u 1:(100*$2) t "Coronary" w lines lw 8 lt 2 smooth acsplines, "auc_time6" u 1:(100*$4) t "Medical" w lines lw 8 smooth acsplines, "auc_time6" u 1:(100*$5) t "Surgical" w lines lw 8 smooth acsplines
