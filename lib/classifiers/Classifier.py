import time
from lib.utils import format_runtime, plot_scores
from lib.evaluate.StandardEval import my_eval

from sklearn.model_selection import learning_curve


class Classifier:
    """
    Generic Classifier that performs recurring machine learning tasks
    """
    def __init__(self, model, n_epochs, logger, patience, fig_dir, model_name, print_every):
        self.model = model
        self.n_epochs = n_epochs
        self.logger = logger
        self.patience = patience
        self.fig_dir = fig_dir
        self.model_name = model_name
        self.print_every = print_every

        # empty now and set during or after training
        self.train_time = 0
        self.prev_val_f1 = 0
        self.best_val_f1 = 0
        self.full_patience = patience
        self.current_patience = self.full_patience
        self.test_perf = []
        self.test_perf_string = ''

    def train_epoch(self, train_batches):
        start = time.time()
        epoch_loss = 0
        for step, batch in enumerate(train_batches):

            loss = self.model.train_on_batch(batch)

            if (step > 0) & (step % self.print_every == 0):
                self.logger.info(f' > Step {step}/{len(train_batches)}: loss = {loss}')

            epoch_loss += loss
        elapsed = time.time() - start
        elapsed = format_runtime(elapsed)
        return epoch_loss, elapsed

    def predict_eval(self, batches, labels):
        preds, loss = self.model.predict(batches)
        return my_eval(labels, preds, av_loss=loss)

    def train_epoch_then_val(self, train_batches, dev_batches, dev_labels):
        epoch_loss, ep_elapsed = self.train_epoch(train_batches)
        val_metrics, val_perf_string = self.predict_eval(dev_batches, dev_labels)

        train_loss = epoch_loss / len(train_batches)
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['f1']

        return train_loss, ep_elapsed, val_loss, val_f1, val_perf_string

    def update_patience(self, val_f1):
        # if an improvement happens, we have full patience, if no improvement happens
        # patience goes down, if patience reaches zero, we stop training
        if val_f1 > self.prev_val_f1:
            self.current_patience = self.full_patience
        else:
            self.current_patience -= 1

    def train_all_epochs(self, train_batches, dev_batches, dev_lables):
        self.model.model.train()

        train_start = time.time()
        losses = []
        for ep in range(self.n_epochs):
            # ep = self.start_epoch + ep
            train_loss, ep_elapsed, val_loss, val_f1, val_perf_string = self.train_epoch_then_val(train_batches,
                                                                                                  dev_batches,
                                                                                                  dev_lables)
            losses.append((train_loss, val_loss))
            self.logger.info(f' > {self.model_name.upper()} Epoch {ep} (took {ep_elapsed}): Av loss = {train_loss}, '
                             f'Val perf: {val_perf_string} (Best f1 so far: {self.best_val_f1})')

            # if save:
            #    self.model.save_bert_model(self.model, self.cp_dir, f'epoch{ep}')

            if val_f1 > self.prev_val_f1:
                self.current_patience = self.full_patience
            else:
                self.current_patience -= 1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1

            self.update_patience(val_f1)

            self.prev_val_f1 = val_f1

            if not self.current_patience > 0:
                if val_f1 > 0.20:
                    self.logger.info(" Stopping training.")
                    break

        eps_elapsed = time.time() - train_start
        return eps_elapsed, losses

    def train_on_fold(self, fold):
        train_elapsed, losses = self.train_all_epochs(fold['train_batches'],
                                                      fold['dev_batches'],
                                                      fold['dev'].label)

        loss_plt = plot_scores(losses)
        loss_plt.savefig(self.fig_dir + f'/{self.model_name}_trainval_loss.png', bbox_inches='tight')

        self.train_time = format_runtime(train_elapsed)

        # test model
        test_metrics, test_perf_string = self.predict_eval(fold['test_batches'], fold['test'].label)
        acc, prec, rec, f1 = [test_metrics['acc'], test_metrics['prec'], test_metrics['rec'], test_metrics['f1']]

        self.test_perf = [acc, prec, rec, f1]
        self.test_perf_string = test_perf_string

        self.logger.info(f' Model {self.model_name} on Fold {fold["name"]} (took {self.train_time}): Test perf: {self.test_perf_string}')

        #cp_fp = format_checkpoint_filepath(self.cp_dir, bertcam='bert', epoch_number=ep)
        #save_bert_model(self.model, cp_fp)
