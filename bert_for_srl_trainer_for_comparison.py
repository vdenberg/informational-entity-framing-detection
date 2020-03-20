# This training code is based on the script here: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
import random, time, os
import torch
import numpy as np
import bert_models.utils_bert as ub
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import logging, sys, argparse
from transformers import BertTokenizer


if __name__ == "__main__":
    """
    RUN EXAMPLE:
        python bert_models/bert_for_srl_trainer.py --train_path datasets/X-SRL_Test/mini_X-SRL_Test_EN.json \
        --dev_path datasets/X-SRL_Test/mini_X-SRL_Test_EN.json -s saved_models/TEST_BERT \
        -b "bert-base-uncased" -ep 4 -bs 2 -inf 2 
        
        /opt/slurm/bin/srun --partition gpulong --gres=gpu:1 --mem 16GB \
        python bert_models/bert_for_srl_trainer.py \
        --train_path /home/mitarb/daza/datasets/X-SRL/JSON/TRAIN-BERT-SCOREv1/ES_X-SRL_train.bscore.json \
        --dev_path /home/mitarb/daza/datasets/X-SRL/JSON/TRAIN-BERT-SCOREv1/ES_X-SRL_dev.bscore.json \
        --save_model_dir saved_models/BERT_FINETUNE_TEST_ES \
        --epochs 20 --batch_size 16 -mx 256 -inf 100 -lr 0.0001 
    """

    # =====================================================================================
    #                    GET PARAMETERS
    # =====================================================================================
    # Read arguments from command line
    parser = argparse.ArgumentParser()

    # GENERAL SYSTEM PARAMS
    parser.add_argument('-t', '--train_path', help='Filepath containing the Training JSON', required=True)
    parser.add_argument('-d', '--dev_path', help='Filepath containing the Validation JSON', required=True)
    parser.add_argument('-s', '--save_model_dir', required=True)
    parser.add_argument('-b', '--bert_model', default="bert-base-multilingual-cased")

    # NEURAL NETWORK PARAMS
    parser.add_argument('-sv', '--seed_val', type=int, default=1373)
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-inf', '--info_every', type=int, default=10)
    parser.add_argument('-mx', '--max_len', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5)
    parser.add_argument('-gr', '--gradient_clip', type=float, default=1.0)


    args = parser.parse_args()

    TRAIN_DATA_PATH = args.train_path
    DEV_DATA_PATH = args.dev_path
    MODEL_DIR = args.save_model_dir
    LOSS_FILENAME = f"{MODEL_DIR}/Losses.json"
    LABELS_FILENAME = f"{MODEL_DIR}/label2index.json"

    BERT_MODEL_NAME = args.bert_model
    DO_LOWERCASE = False

    SEED_VAL = args.seed_val
    SEQ_MAX_LEN = args.max_len
    EPOCHS = args.epochs
    PRINT_INFO_EVERY = args.info_every
    GRADIENT_CLIP = args.gradient_clip
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    # =====================================================================================
    #                    LOGGING INFO ...
    # =====================================================================================
    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"{MODEL_DIR}/BERT_TokenClassifier.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
    logging.info("Start Logging")
    logging.info(args)

    # If there's a GPU available...
    device, USE_CUDA = ub.get_torch_device()
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # ==========================================================================================
    #               LOAD TRAIN & DEV DATASETS
    # ==========================================================================================
    label2index = ub.build_label_vocab(TRAIN_DATA_PATH)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=DO_LOWERCASE)
    train_inputs, train_masks, train_labels = ub.load_srl_dataset(TRAIN_DATA_PATH, tokenizer,
                                                                  label2index, max_len=SEQ_MAX_LEN)
    ub.save_label_dict(label2index, filename=LABELS_FILENAME)

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    dev_inputs, dev_masks, dev_labels = ub.load_srl_dataset(DEV_DATA_PATH, tokenizer,
                                                            label2index, max_len=SEQ_MAX_LEN)

    # Create the DataLoader for our Development set.
    dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)

    # ==========================================================================================
    #              LOAD MODEL & OPTIMIZER
    # ==========================================================================================
    model = BertForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(label2index))
    if USE_CUDA: model.cuda()

    optimizer = AdamW(model.parameters(), lr =LEARNING_RATE, eps = 1e-8)

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * EPOCHS

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)


    # ==========================================================================================
    #                          TRAINING ...
    # ==========================================================================================

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, EPOCHS):
        # Perform one full pass over the training set.
        logging.info("")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        logging.info('Training...')

        t0 = time.time()
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Progress update
            if step % PRINT_INFO_EVERY == 0 and step != 0:
                # Calculate elapsed time in minutes.
                elapsed = ub.format_time(time.time() - t0)
                # Report progress.
                logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {.4f}.'.format(step,
                                                                                                  len(train_dataloader),
                                                                                                  elapsed,
                                                                                                  loss.item()))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        logging.info("")
        logging.info("  Average training loss: {0:.4f}".format(avg_train_loss))
        logging.info("  Training Epoch took: {:}".format(ub.format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        logging.info("")
        logging.info("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in dev_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            output_vals = torch.softmax(logits, dim=1)

            # Move logits and labels to CPU
            output_vals = output_vals.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = ub.flat_accuracy(output_vals, label_ids)
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy
            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        logging.info("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        logging.info("  Validation took: {:}".format(ub.format_time(time.time() - t0)))

        # ================================================
        #               Save Checkpoint for this Epoch
        # ================================================
        ub.save_model(f"{MODEL_DIR}/EPOCH_{epoch_i}", {"args":[]}, model, tokenizer)

    ub.save_losses(loss_values, filename=LOSS_FILENAME)

    logging.info("")
    logging.info("Training complete!")