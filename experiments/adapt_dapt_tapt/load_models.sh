python -m download_model \
        --model allenai/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_5015 \
        --serialization_dir $(pwd)/pretrained_models/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_5015

python -m download_model \
        --model allenai/dsp_roberta_base_tapt_hyperpartisan_news_5015 \
        --serialization_dir $(pwd)/pretrained_models/dsp_roberta_base_tapt_hyperpartisan_news_5015

python -m download_model \
        --model allenai/news_roberta_base \
        --serialization_dir $(pwd)/pretrained_models/news_roberta_base

python convert_vocab.py
python adapt_config.py

#allenai/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_5015
#allenai/dsp_roberta_base_dapt_news_tapt_hyperpartisan_news_515
#allenai/dsp_roberta_base_tapt_hyperpartisan_news_5015
#allenai/dsp_roberta_base_tapt_hyperpartisan_news_515