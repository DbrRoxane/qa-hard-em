data=$1
loss=$2
bs=32
pbs=512
data_dir="preprocessed-open-domain-qa-data"

if [ ${loss} = "hard-em" ]
then
    tau=$3
    output_dir="out/${data}-${loss}-${tau}-${bs}-${pbs}"
else
    output_dir="out/${data}-${loss}-${bs}-${pbs}"
    tau=0
fi

#train_file="${data_dir}/${data}-train0.json"
#for index in 1 2 3 ; do
#    train_file="${train_file},${data_dir}/${data}-train${index}.json"
#done
#dev_file="${data_dir}/${data}-dev.json"
#test_file="${data_dir}/${data}-test.json"
train_file="../data/narrativeqa/min_summaries_with_answer_train.json2"
dev_file="../data/narrativeqa/min_summaries_with_answer_dev.json2"
test_file="../data/narrativeqa/min_summaries_with_answer_test.json"



python3 main.py --do_train --output_dir ${output_dir} \
          --train_file ${train_file} --predict_file ${dev_file} \
          --train_batch_size ${bs} --predict_batch_size ${pbs} --verbose --loss_type ${loss} --tau ${tau} --verbose_logging 

#python3 main.py --do_predict --output_dir ${output_dir} \
#          --predict_file ${dev_file} \
#          --init_checkpoint ${output_dir}/best-model.pt \
#          --predict_batch_size ${pbs} --n_paragraphs "10,20,40,80" --prefix dev_
#python3 main.py --do_predict --output_dir ${output_dir} \
#          --predict_file ${test_file} \
#          --init_checkpoint ${output_dir}/best-model.pt \
#          --predict_batch_size ${pbs} --n_paragraphs "10,20,40,80" --prefix test_


