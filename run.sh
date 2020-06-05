data=$1
loss=$2
bs=16
pbs=256

P1_dir="../data/output/P1_SUM/"
P2_dir="../data/output/P2_CRCAE/"
P3_dir="../data/output/P3_CRNAE/"

P1_ckpt="${P1_dir}best-model.pt"
P2_ckpt="${P2_dir}best-model.pt"
P3_ckpt="${P3_dir}best-model.pt"

data_dir="../data/narrativeqa/min_final"

if [ ${loss} = "hard-em" ]
then
    tau=$3
else
    tau=0
fi

P1_train="${data_dir}SUM_r6_train.json"
P2_train="${data_dir}CRCAE_bertbm25_3para_r6_train.json"
P3_train="${data_dir}CRNAE_bertbm25_3para_r6_train.json"

sum_dev="${data_dir}SUM_r6_dev.json"
sum_test="${data_dir}SUM_r6_test.json"
CRCAE_dev="${data_dir}CRCAE_allranking_3para_r5_dev.json"
CRCAE_test="${data_dir}CRCAE_allranking_3para_r5_test.json"
CRNAE_dev="${data_dir}CRNAE_allranking_3para_r5_dev.json"
CRNAE_test="${data_dir}CRNAE_allranking_3para_r5_test.json"
NRNAE_dev="${data_dir}NRNAE_r6_20para_dev.json"
NRNAE_test="${data_dir}NRNAE_r6_20para_test.json"

python3 main.py --do_predict --output_dir ${P1_dir} \
          --predict_file ${CRCAE_test} \
	  --init_checkpoint ${P1_ckpt}\
          --predict_batch_size ${pbs} --n_paragraphs "6,9,12" --prefix CRCAE_allranking_3para_r5_test_
python3 main.py --do_predict --output_dir ${P1_dir} \
          --predict_file ${CRCAE_dev} \
	  --init_checkpoint ${P1_ckpt}\
          --predict_batch_size ${pbs} --n_paragraphs "6,9,12" --prefix CRCAE_allranking_3para_r5_dev_

python3 main.py --do_predict --output_dir ${P2_dir} \
          --predict_file ${NRNAE_test} \
	  --init_checkpoint ${P2_ckpt}\
          --predict_batch_size ${pbs} --n_paragraphs "10,15,20" --prefix NRNAE_BERT_20para_r5_test_
python3 main.py --do_predict --output_dir ${P2_dir} \
          --predict_file ${NRNAE_dev} \
	  --init_checkpoint ${P2_ckpt}\
          --predict_batch_size ${pbs} --n_paragraphs "10,15,20" --prefix NRNAE_BERT_20para_r5_dev_

python3 main.py --do_predict --output_dir ${P3_dir} \
          --predict_file ${NRNAE_dev} \
	  --init_checkpoint ${P3_ckpt}\
          --predict_batch_size ${pbs} --n_paragraphs "10,15,20" --prefix NRNAE_BERT_20para_r5_dev_
python3 main.py --do_predict --output_dir ${P3_dir} \
          --predict_file ${NRNAE_test} \
	  --init_checkpoint ${P3_ckpt}\
          --predict_batch_size ${pbs} --n_paragraphs "10,15,20" --prefix NRNAE_BERT_20para_r5_test_

python3 main.py --do_predict --output_dir ${P3_dir} \
          --predict_file ${CRCAE_dev} \
	  --init_checkpoint ${P3_ckpt}\
          --predict_batch_size ${pbs} --n_paragraphs "6,9,12" --prefix CRCAE_allranking_3para_r5_dev_
python3 main.py --do_predict --output_dir ${P3_dir} \
          --predict_file ${CRCAE_test} \
	  --init_checkpoint ${P3_ckpt}\
          --predict_batch_size ${pbs} --n_paragraphs "6,9,12" --prefix CRCAE_allranking_3para_r5_test_


#train_file="${data_dir}/${data}-train0.json"
#for index in 1 2 3 ; do
#    train_file="${train_file},${data_dir}/${data}-train${index}.json"
#done
#dev_file="${data_dir}/${data}-dev.json"
#test_file="${data_dir}/${data}-test.json"
#train_file="${data_dir}/../min_all_with_answer_train.json_bertbm25oracle_r6_3para"
#min_sums_train.json_r6"
#dev_file_all_with="${data_dir}/../min_all_with_answer_dev.json_allrankingtech_r5_3para"
#min_all_without_answer_dev.json_realwithout_r6"
#test_file_all_with="${data_dir}/../min_all_without_answer_test.json_allrankingtech_severalanswers_r5_28mai"
#min_all_without_answer_test.json_realwithout_r6"
##min_all_with_answer_test.json_allrankingtech_severalanswers_r5_28mai"
#dev_file_all_without="${data_dir}/../min_all_without_answer_dev.json_realwithout_r6"
#test_file_all_without="${data_dir}/../min_all_without_answer_test.json_realwithout_r6"

#dev_file_sum="${data_dir}/min_sums_dev.json_r6"
#test_file_sum="${data_dir}/min_sums_test.json_r6"

#crcae_dev="${data_dir}/../min_all_with_answer_dev.json_allrankingtech_r5_3para_oracle"
#crcae_test="${data_dir}/../min_all_without_answer_test.json_allrankingtech_severalanswers_r5_28mai_oracle"

#crnae_dev="${data_dir}/../min_all_with_answer_dev.json_allrankingtech_r5_3para"
#crnae_test="${data_dir}/../min_all_without_answer_test.json_allrankingtech_severalanswers_r5_28mai"

#nrnae_dev="${data_dir}/../min_all_without_answer_dev.json_realwithout_r6"
#nrnae_test="${data_dir}/../min_all_without_answer_test.json_realwithout_r6"


#min_sums_with_answer_test.json_r5"

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

#python3 main.py --do_train --output_dir ${output_dir} \
#          --init_checkpoint ${init_ckpt} \
#	  --train_file ${train_file} --predict_file ${dev_file_all_with} \
#          --train_batch_size ${bs} --predict_batch_size ${pbs} --verbose --loss_type ${loss} --tau ${tau} --verbose_logging #

#python3 main.py --do_predict --output_dir ${output_dir} \
#         --predict_file ${test_file_sum} \
#	  --init_checkpoint "${output_dir}/best-model.pt"\
#          --predict_batch_size ${pbs} --n_paragraphs "1" --prefix min_predsum_r6_test
#
#python3 main.py --do_predict --output_dir ${output_dir} \
#          --predict_file ${dev_file_sum} \
#	  --init_checkpoint "${output_dir}/best-model.pt" \
#          --predict_batch_size ${pbs} --n_paragraphs "1" --prefix min_predsum_r6_dev

#python3 main.py --do_predict --output_dir ${P2_dir} \
#          --predict_file ${crcae_test} \
#	  --init_checkpoint ${P2_ckpt}\
#          --predict_batch_size ${pbs} --n_paragraphs "1,5,10,15,20" --prefix trainp2_test_crcae
#python3 main.py --do_predict --output_dir ${P2_dir} \
#          --predict_file ${crcae_dev} \
#	  --init_checkpoint ${P2_ckpt}\
#          --predict_batch_size ${pbs} --n_paragraphs "1,5,10,15,20" --prefix trainp2_dev_crcae

#python3 main.py --do_predict --output_dir ${output_dir} \
#          --predict_file ${dev_file_all_without} \
#	  --init_checkpoint "${output_dir}/best-model.pt" \
#          --predict_batch_size ${pbs} --n_paragraphs "1,5,10,15,20" --prefix min_predall_without_r6_20para_dev

#
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
