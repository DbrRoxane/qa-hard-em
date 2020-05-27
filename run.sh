data=$1
loss=$2
bs=16
pbs=256
output_dir="../data/output/min_predictions_sum_r5_3passages_BERT/"
data_dir="../data/narrativeqa/min_format"

if [ ${loss} = "hard-em" ]
then
    tau=$3
    #output_dir= "/nas-data/trivia-qa/outputs/trivia-qa-job-0.1.dev53-ga79f2d5/narrativeqa-first-only_14mai/"
else
    #output_dir="""${output_base_folder}/${data}-${loss}"
    #"out/${data}-${loss}-${bs}-${pbs}"
    tau=0
fi

#train_file="${data_dir}/${data}-train0.json"
#for index in 1 2 3 ; do
#    train_file="${train_file},${data_dir}/${data}-train${index}.json"
#done
#dev_file="${data_dir}/${data}-dev.json"
#test_file="${data_dir}/${data}-test.json"
train_file="${data_dir}/min_sums_with_answer_train.json_r5"
dev_file="${data_dir}/min_sums_with_answer_dev.json_r5"
test_file="${data_dir}/min_sums_with_answer_test.json_r5"

#train_file="${data_dir}/min_summaries_with_answer_train.json"
#all_with_dev_file5="${data_dir}/min_all_with_answer_dev.json_r5"
#all_with_dev_file6="${data_dir}/min_all_with_answer_dev.json_r6"
#all_with_dev_file7="${data_dir}/min_all_with_answer_dev.json_r7"
#all_with_test_file="${data_dir}/min_all_with_answer_test.json_several_answers_r5_5parag"
#min_all_with_answer_test.json"
#all_without_dev_file="${data_dir}/min_all_without_answer_dev.json"
#all_without_test_file="${data_dir}/min_all_without_answer_test.json" #/min_summaries_with_answer_test.json"
#sum_test_file="${data_dir}/min_summaries_without_answer_test.json"
#sum_dev_file="${data_dir}/min_summaries_without_answer_dev.json"

python3 main.py --do_train --output_dir ${output_dir} \
          --train_file ${train_file} --predict_file ${dev_file} \
          --train_batch_size ${bs} --predict_batch_size ${pbs} --verbose --loss_type ${loss} --tau ${tau} --verbose_logging 
#
#python3 /app/trivia-qa/main.py --do_predict --output_dir ${output_dir} \
#          --predict_file ${all_with_test_file} \
#	  --init_checkpoint "/nas-data/trivia-qa/outputs/trivia-qa-job-0.1.dev53-ga79f2d5/narrativeqa-first-only_14mai/best-model.pt"\
#          --predict_batch_size ${pbs} --n_paragraphs "10,20,40,80" --prefix min_predallwithanswer_trainedsumwithanswer_test
#	  --init_checkpoint "/nas-data/trivia-qa/outputs/trivia-qa-job-0.1.dev53-ga79f2d5/narrativeqa-first-only_14mai/best-model.pt"\

#python3 main.py --do_predict --output_dir ${output_dir} \
#          --predict_file ${all_with_test_file} \
#	  --init_checkpoint "narrativeqa_summary_3passages_r5_firstonly_bestmodel.pt" \
#          --predict_batch_size ${pbs} --n_paragraphs "1,2,3,4,5" --prefix "min_all_with_answer_test.json_several_answers_r5_5parag"
#python3 /app/trivia-qa/main.py --do_predict --output_dir ${output_dir} \
#          --predict_file ${all_without_test_file} \
#	  --init_checkpoint "/nas-data/trivia-qa/outputs/trivia-qa-job-0.1.dev53-ga79f2d5/narrativeqa-first-only_14mai/best-model.pt"\
#          --predict_batch_size ${pbs} --n_paragraphs "10,20,40,80" --prefix min_predallwithoutanswer_trainedsumwithanswer_test
#	  --init_checkpoint "/nas-data/trivia-qa/outputs/trivia-qa-job-0.1.dev53-ga79f2d5/narrativeqa-first-only_14mai/best-model.pt"\

#python3 /app/trivia-qa/main.py --do_predict --output_dir ${output_dir} \
#          --predict_file ${all_without_dev_file} \
#	  --init_checkpoint "/nas-data/trivia-qa/outputs/trivia-qa-job-0.1.dev53-ga79f2d5/narrativeqa-first-only_14mai/best-model.pt"\
#          --predict_batch_size ${pbs} --n_paragraphs "10,20,40,80" --prefix min_predallwithoutanswer_trainedsumwithanswer_dev

#python3 /app/trivia-qa/main.py --do_predict --output_dir ${output_dir} \
#          --predict_file ${sum_dev_file} \
#	  --init_checkpoint "/nas-data/trivia-qa/outputs/trivia-qa-job-0.1.dev53-ga79f2d5/narrativeqa-first-only_14mai/best-model.pt"\
#          --predict_batch_size ${pbs} --n_paragraphs "10,20,40,80" --prefix min_predsum_trainedsumwithanswer_dev
#python3 /app/trivia-qa/main.py --do_predict --output_dir ${output_dir} \
#          --predict_file ${sum_test_file} \
#	  --init_checkpoint "/nas-data/trivia-qa/outputs/trivia-qa-job-0.1.dev53-ga79f2d5/narrativeqa-first-only_14mai/best-model.pt"\
#          --predict_batch_size ${pbs} --n_paragraphs "10,20,40,80" --prefix min_predsum_trainedsumwithanswer_test
          #--init_checkpoint ${output_dir}/best-model.pt \
