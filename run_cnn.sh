
### <- set paths - >


data="./final_data/"

CNN_SCR="cnn_ext_coherence.py"
#EXP_DIR="saved_exp/"
MODEL_DIR="saved_CNET/"

mkdir -p $MODEL_DIR
#mkdir -p $EXP_DIR

###<- Set general DNN settings ->
dr_ratios=(0.5) #dropout_ratio
mb_sizes=(64 32) #minibatch-size

### <- set CNN settings ->
nb_filters=(150) #no of feature map
w_sizes=(5 3)
pool_lengths=(6 8)
max_lengths=(14000 8000)
emb_sizes=(100 50)




log="log.CNET"
echo "Training...!" > $log


#for feat in ${features[@]}; do
	
for ratio in ${dr_ratios[@]}; do
	for nb_filter in ${nb_filters[@]}; do
		for w_size in ${w_sizes[@]}; do
			for pool_len in ${pool_lengths[@]}; do
				for mb in ${mb_sizes[@]}; do
					for maxlen in ${max_lengths[@]}; do
						for emb_size in ${emb_sizes[@]}; do

							echo "INFORMATION: dropout_ratio=$ratio filter-nb=$nb_filter w_size=$w_size pool_len=$pool_len batch-size=$mb maxlen=$maxlen emb_size=$emb_size feats=$feat">> $log;
							echo "----------------------------------------------------------------------" >> $log;

							THEANO_FLAGS=device=gpu0,floatX=float32 python $CNN_SCR --data-dir=$data --model-dir=$MODEL_DIR \
							--dropout_ratio=$ratio --minibatch-size=$mb --emb-size=$emb_size\
							--nb_filter=$nb_filter --w_size=$w_size --pool_length=$pool_len\
							--max-length=$maxlen  >>$log
							wait

							echo "----------------------------------------------------------------------" >> $log;
						done
					done
				done
			done 
		done	
	done 
done
#done
