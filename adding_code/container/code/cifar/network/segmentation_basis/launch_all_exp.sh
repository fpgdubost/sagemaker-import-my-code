for i in $(seq 1 9);
do
    python train.py $((140+$i)) $i
    python test.py $((150+$i)) $((140+$i))
done

