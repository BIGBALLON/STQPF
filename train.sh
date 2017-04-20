#!/bin/bash
for k in $( seq 4 9 )
do
    python3 train_s1.py test${k}.h5 ${k}
    python3 train_s2.py test${k}.h5 ${k}
    python3 predict.py test${k}.h5 > ans.txt
    ./outputanswer ans.txt ans${k}.txt
done



