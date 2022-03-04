#!/bin/bash

log_path="/home/agois/longformer_stuff/entmax-roberta-maia/log.baseline.txt"
shortlog_path="/home/agois/longformer_stuff/entmax-roberta-maia/log_short.baseline.txt"

if [ -f $log_path ];then
  echo "log file already exists! Exiting..."
  exit 1
fi
if [ -f $shortlog_path ];then
  echo "short log file already exists! Exiting..."
  exit 1
fi

lr=.01

#window-only baseline
for window in 3 5 7 9 11 15 19 23 25 30 40 50 75 100 125 150 175 200 250 300 350 400 450 500 512
do
  echo "NEW RUN --- only window:$window " >> $log_path
  python extender/quantization.py \
    --window $window\
    --dist cos\
    --dist-t 0\
    --lr $lr \
    --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
    --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_extender_temp.pickle\
    >> $log_path
    tail -n1 $log_path >> $shortlog_path
done

# kmeans
for bsize in 3 4 5 8
do
  echo "NEW RUN --- no-window kmeans --proj-dim 8 -k $bsize" >> $log_path
  python extender/quantization.py \
    -s $bsize\
    --lr $lr \
    --grouping kmeans \
    --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
    --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_dim8.pickle\
    -r 8\
  >> $log_path
  tail -n1 $log_path >> $shortlog_path
done

for dim in 16 32
do
  echo "NEW RUN --- no-window kmeans --proj-dim $dim -k 2" >> $log_path
    python extender/quantization.py \
      -s 2\
      --lr $lr \
      --grouping kmeans \
      --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
      --save /home/agois/longformer_stuff/entmax-roberta-maia/projections_dim${dim}.pickle\
      -r $dim\
    >> $log_path
    tail -n1 $log_path >> $shortlog_path
  for bsize in 3 4 5 8
  do
    echo "NEW RUN --- no-window kmeans --proj-dim $dim -k $bsize" >> $log_path
    python extender/quantization.py \
      -s $bsize\
      --lr $lr \
      --grouping kmeans \
      --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
      --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_dim${dim}.pickle\
      -r $dim\
    >> $log_path
    tail -n1 $log_path >> $shortlog_path
  done
done
for window in 3 5 7 9 11 15 19 23 25
do
  for bsize in 2 3 4 5 8
  do
    for n in 1 2 3 4 5
    do
      echo "NEW RUN --- window:$window kmeans --cluster-rounds $n -k $bsize" >> $log_path
      python extender/quantization.py \
        -s $bsize\
        --window $window\
        --lr $lr \
        --grouping kmeans \
        --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
        --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_extender_temp.pickle\
        --cluster-rounds $n\
        >> $log_path
        tail -n1 $log_path >> $shortlog_path
    done
  done
done


# quantization - fixed
for bsize in 4 8 12 16 20
do
  echo "NEW RUN --- no-window --same-size -s $bsize" >> $log_path
  python extender/quantization.py \
    -s $bsize\
    --lr $lr \
    --same-size \
    --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
    --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_extender_temp.pickle\
  >> $log_path
  tail -n1 $log_path >> $shortlog_path
done
for window in 3 5 7 9 11 15 19 23 25
do
  for bsize in 4 8 12 16 20
  do
    echo "NEW RUN --- window:$window --same-size -s $bsize" >> $log_path
    python extender/quantization.py \
      -s $bsize\
      --window $window\
      --lr $lr \
      --same-size \
      --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
      --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_extender_temp.pickle\
      >> $log_path
      tail -n1 $log_path >> $shortlog_path
  done
done

#quantization - dynamic
for bsize in 4 8 12 16 20
do
  echo "NEW RUN --- no-window -s $bsize" >> $log_path
  python extender/quantization.py \
    -s $bsize\
    --lr $lr \
    --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
    --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_extender_temp.pickle\
  >> $log_path
  tail -n1 $log_path >> $shortlog_path
done
for window in 3 5 7 9 11 15 19 23 25
do
  for bsize in 4 8 12 16 20
  do
    echo "NEW RUN --- window:$window -s $bsize" >> $log_path
    python extender/quantization.py \
      -s $bsize\
      --window $window\
      --lr $lr \
      --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
      --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_extender_temp.pickle\
      >> $log_path
      tail -n1 $log_path >> $shortlog_path
  done
done

#distance-based projections
for dist_t in .2 .4 .6 .8 1.0
do
  echo "NEW RUN --- no-window dist_t:$dist_t" >> $log_path
  python extender/quantization.py \
    --dist cos\
    --dist-t $dist_t\
    --lr $lr \
    --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
    --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_extender_temp.pickle\
    >> $log_path
    tail -n1 $log_path >> $shortlog_path
done
for window in 3 5 7 9 11 15 19 23 25
do
  for dist_t in .2 .4 .6 .8 1.0
  do
    echo "NEW RUN --- window:$window dist_t:$dist_t" >> $log_path
    python extender/quantization.py \
      --window $window\
      --dist cos\
      --dist-t $dist_t\
      --lr $lr \
      --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
      --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_extender_temp.pickle\
      >> $log_path
      tail -n1 $log_path >> $shortlog_path
  done
done

# rand-projections+window (big-bird)
for dist_t in .2 .4 .6 .8 1.0
do
  echo "NEW RUN --- big bird - no-window dist_t:$dist_t" >> $log_path
  python extender/quantization.py \
    --dist cls_random\
    --dist-t $dist_t\
    --lr $lr \
    --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
    --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_extender_temp.pickle\
    >> $log_path
    tail -n1 $log_path >> $shortlog_path
done
for window in 3 5 7 9 11 15 19 23 25
do
  for dist_t in .2 .4 .6 .8 1.0
  do
    echo "NEW RUN --- big bird - window:$window dist_t:$dist_t" >> $log_path
    python extender/quantization.py \
      --window $window\
      --dist cls_random\
      --dist-t $dist_t\
      --lr $lr \
      --data /home/agois/longformer_stuff/entmax-roberta-maia/kqs_enc-attn.pt \
      --load /home/agois/longformer_stuff/entmax-roberta-maia/projections_extender_temp.pickle\
      >> $log_path
      tail -n1 $log_path >> $shortlog_path
  done
done