declare -i cnt=1
while read line 
do 
    youtube-dl  -x --audio-format wav -o "audios/(seq -f '%02g' $cnt)-%(title)s.%(ext)s" $line
    cnt=($cnt+1)
    sleep 2 
done < ../links.txt