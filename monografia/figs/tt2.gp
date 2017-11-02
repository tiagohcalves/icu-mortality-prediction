set terminal postscript eps enhanced color font 'Helvetica,20'

set bar 1.000000 front
set border 3 front lt black linewidth 1.000 dashtype solid
set boxwidth 0.8 absolute
set style fill   solid 1.00 noborder
set grid noxtics nomxtics ytics nomytics noztics nomztics \
 nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics
set grid layerdefault   lt 0 linewidth 0.500,  lt 0 linewidth 0.500
set key bmargin center horizontal Left reverse noenhanced autotitle columnhead nobox
set style histogram rowstacked title textcolor lt -1 offset character 2, 0.25, 0
set style textbox transparent margins  1.0,  1.0 border
unset logscale
set datafile missing '-'
set style data histograms
set title "" 
set xlabel "" 
set ylabel "Numero de publicacoes" 
set colorbox vertical origin screen 0.9, 0.2, 0 size screen 0.05, 0.6, 0 front bdefault
set output 'npubs.eps'
plot 'publicacoes' using "Periodico":xtic(1) t col, '' u "Conferencia" t col, '' u "Livro" t col

set ylabel "Numero de citacoes" 
set output 'tcit.eps'
plot 'tcit' u 2:xtic(1) t ""

reset
set key bottom
set xtics border in scale 0,0 nomirror rotate by -90  autojustify
set xtics 1
set ylabel "Numero de citacoes"
#set output 'citp.eps'
#plot 'citp' u 2:1:(log($3+1)) t "Periodico" w points lt 6 ps variable lc "blue", 'citp2' u 2:1:(log($3+1)) t "Conferencia" w points lt 6 ps variable lc "red"

reset
set bar 1.000000 front
set border 3 front lt black linewidth 1.000 dashtype solid
set boxwidth 0.75 absolute
set style fill   solid 1.00 border lt -1
set style circle radius graph 0.02, first 0.00000, 0.00000 
set style ellipse size graph 0.05, 0.03, first 0.00000 angle 0 units xy
set grid nopolar
set grid noxtics nomxtics ytics nomytics noztics nomztics \
 nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics
set grid layerdefault   lt 0 linewidth 0.500,  lt 0 linewidth 0.500
set key outside right top vertical Left reverse noenhanced autotitle columnhead nobox
set key invert samplen 4 spacing 1 width 0 height 0 
set style histogram rowstacked title textcolor lt -1
set style textbox transparent margins  1.0,  1.0 border
set style data histograms
set xtics border in scale 0,0 nomirror rotate by -90  autojustify
unset ytics
set title ""
set ylabel "% do total anual" 
set yrange [ 0.00000 : 100.000 ] noreverse nowriteback
set colorbox vertical origin screen 0.9, 0.2, 0 size screen 0.05, 0.6, 0 front bdefault

set output 'nqual.eps'
plot 'npap' using (100.*$2/$9):xtic(1) t column(2), for [i=3:8] '' using (100.*column(i)/column(9)) title column(i)
