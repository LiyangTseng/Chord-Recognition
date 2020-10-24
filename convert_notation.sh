for i in {1..200}
do 
python3 convert.py ../CE200/$i/ground_truth.txt ../CE200/$i/shorthand_gt.txt shorthand
done
