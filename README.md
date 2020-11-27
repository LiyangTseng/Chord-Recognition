# Chord-Recognition

## File usage
<code>convert_notation.sh:</code> convert all label files to "shorthand" format using <code>convert.py:</code>

<code>download_audios.py:</code> download wav files from Youtube

<code>train.py:</code> train model<br > in Audiodataset: </br>
* --num_workers=20 => multiprocessing
* --preprocessing=False => do not preprocess
* --train=True => use dataset for training

<code>evaluate.py:</code> store test results at <code>score.csv</code> with the following format

<code>separate.py:</code> using spleeter to separate source to tracks, and then merge the chord-related part

