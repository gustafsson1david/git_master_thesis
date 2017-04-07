count=(1 2 3 4 5 6 7 8 9 10 11 12)
for c in "${count[@]}"
do
   python3 slim_inception_v3.py $c
done
