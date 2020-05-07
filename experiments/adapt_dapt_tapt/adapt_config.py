import json, os

models_dir = 'experiments/adapt_dapt_tapt/pretrained_models'
for model_dir in os.listdir(models_dir):
    config_fp = os.path.join(models_dir, model_dir, 'config.json')
    bert_config_fp = os.path.join(models_dir, model_dir, 'bert_config.json')

    # load config
    config = json.load(open(config_fp, encoding='utf-8'))
    config["eos_token_id"] = 102

    with open(bert_config_fp, 'w') as f:
        json.dump(config, f, indent=True)
