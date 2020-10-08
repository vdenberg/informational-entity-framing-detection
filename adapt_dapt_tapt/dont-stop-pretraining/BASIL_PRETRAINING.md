# Adaptive Pretraining


## On TPUs

To run adaptive pretraining on TPUs, refer to https://github.com/allenai/tpu-pretrain

## On GPUs

To run adaptive pretraining on GPUs, use the `run_language_modeling.py` example from the huggingface repository, which we have copied over to `scripts/run_language_modeling.py`.

Just supply an input file with newline separated documents, e.g. `input.txt`


### DAPT
```
python -m scripts.run_language_modeling --train_data_file tweets.txt \
                                        --line_by_line \
                                        --output_dir roberta-twitter-dapt \
                                        --model_type roberta-base \
                                        --tokenizer_name roberta-base \
                                        --mlm \
                                        --per_gpu_train_batch_size 16 \
                                        --gradient_accumulation_steps 128  \
                                        --model_name_or_path roberta-base \
                                        --do_train \
                                        --max_steps 12500  \
                                        --learning_rate 0.0005
```

### TAPT
```
/opt/slurm/bin/srun --partition gpushort --gres=gpu:1 --mem 20GB python -m scripts.run_language_modeling --train_data_file basil_train.txt \
                                        --line_by_line \
                                        --output_dir roberta-tapt \
                                        --model_type roberta-base \
                                        --tokenizer_name roberta-base \
                                        --mlm \
                                        --per_gpu_train_batch_size 6 \
                                        --gradient_accumulation_steps 6  \
                                        --model_name_or_path roberta-base \
                                        --do_eval \
                                        --eval_data_file basil_dev.txt \
                                        --evaluate_during_training  \
                                        --do_train \
                                        --num_train_epochs 100  \
                                        --learning_rate 0.0001 \
                                        --logging_steps 50
```
### DAPTTAPT
```
/opt/slurm/bin/srun --partition gpushort --gres=gpu:1 --mem 20GB python -m scripts.run_language_modeling --train_data_file basil_train.txt \
                                        --line_by_line \
                                        --output_dir roberta-dapttapt \
                                        --model_type roberta-base \
                                        --tokenizer_name roberta-base \
                                        --mlm \
                                        --per_gpu_train_batch_size 6 \
                                        --gradient_accumulation_steps 6  \
                                        --model_name_or_path ../pretrained_models/news_roberta_base \
                                        --do_eval \
                                        --eval_data_file basil_dev.txt \
                                        --evaluate_during_training  \
                                        --do_train \
                                        --num_train_epochs 100  \
                                        --learning_rate 0.0001 \
                                        --logging_steps 50
```