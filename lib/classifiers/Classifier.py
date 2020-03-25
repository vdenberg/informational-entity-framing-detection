import time
from lib.utils import format_runtime, plot_scores
from lib.evaluate.StandardEval import my_eval

from sklearn.model_selection import learning_curve


class Classifier:
    """
    Generic Classifier that performs recurring machine learning tasks
    """
    def __init__(self, model, n_epochs, logger, patience, fig_dir, model_name, print_every):
        self.wrapper = model
        self.n_epochs = n_epochs
        self.logger = logger
        self.patience = patience
        self.fig_dir = fig_dir
        self.model_name = model_name.upper()
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

            loss = self.wrapper.train_on_batch(batch)
            epoch_loss += loss

            if (step > 0) & (step % self.print_every == 0):
                self.logger.info(f' > Step {step}/{len(train_batches)}: loss = {epoch_loss/step}')

        av_epoch_loss = epoch_loss / len(train_batches)

        elapsed = format_runtime(time.time() - start)
        return av_epoch_loss, elapsed

    def predict_eval(self, batches, labels):
        preds, loss = self.wrapper.predict(batches)
        return my_eval(labels, preds, av_loss=loss)

    def update_patience(self, val_f1):
        # if an improvement happens, we have full patience, if no improvement happens
        # patience goes down, if patience reaches zero, we stop training
        if val_f1 > self.prev_val_f1:
            self.current_patience = self.full_patience
        else:
            self.current_patience -= 1

    def train_all_epochs(self, train_batches, tr_labels, dev_batches, dev_labels):
        train_start = time.time()

        losses = []
        if self.model_name == 'BERT':
            tr_metrics, tr_perf_string = self.predict_eval(train_batches, tr_labels)
            val_metrics, val_perf_string = self.predict_eval(dev_batches, dev_labels)

            tr_loss = tr_metrics['loss']
            tr_f1 = tr_metrics['f1']
            val_loss = val_metrics['loss']
            val_f1 = val_metrics['f1']

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1

            losses.append((tr_loss, val_loss))

            self.logger.info(f' > {self.model_name} Epoch {-1} (took 0m0s): '
                             f'loss = {tr_loss}, Train perf: {tr_f1}, Val perf: {val_perf_string} '
                             f'(Best f1 so far: {self.best_val_f1})')

        for ep in range(self.n_epochs):
            self.wrapper.model.train()

            av_loss, ep_elapsed = self.train_epoch(train_batches)

            tr_metrics, tr_perf_string = self.predict_eval(train_batches, tr_labels)
            val_metrics, val_perf_string = self.predict_eval(dev_batches, dev_labels)

            tr_loss =  round(tr_metrics['loss'], 5)
            tr_f1 = round(tr_metrics['f1'], 4)
            val_loss = round(val_metrics['loss'], 5)
            val_f1 = round(val_metrics['f1'], 4)

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1

            losses.append((tr_loss, val_loss))
            self.logger.info(f' > {self.model_name} Epoch {ep} (took {ep_elapsed}): '
                             f'loss = {tr_loss}, Train f1: {tr_f1}, Val perf: {val_perf_string} '
                             f'(Best f1 so far: {self.best_val_f1})')

            # if save:
            #    self.model.save_bert_model(self.model, self.cp_dir, f'epoch{ep}')

            if val_f1 > self.prev_val_f1:
                self.current_patience = self.full_patience
            else:
                self.current_patience -= 1

            self.update_patience(val_f1)

            self.prev_val_f1 = val_f1

            if not self.current_patience > 0:
                if val_f1 > 0.20:
                    self.logger.info(" Stopping training.")
                    break

        eps_elapsed = format_runtime(time.time() - train_start)
        return eps_elapsed, losses

    def train_on_fold(self, fold):

        train_elapsed, losses = self.train_all_epochs(fold['train_batches'],
                                                      fold['train'].label,
                                                      fold['dev_batches'],
                                                      fold['dev'].label)
        self.train_time = train_elapsed

        # plot learning curve
        loss_plt = plot_scores(losses)
        loss_plt.savefig(self.fig_dir + f'/{self.model_name}_trainval_loss.png', bbox_inches='tight')


        # test model
        test_metrics, test_perf_string = self.predict_eval(fold['test_batches'], fold['test'].label)
        self.test_perf = [test_metrics['acc'], test_metrics['prec'], test_metrics['rec'], test_metrics['f1']]
        self.test_perf_string = test_perf_string
        self.logger.info(f' Model {self.model_name} on Fold {fold["name"]} (took {self.train_time}): Test perf: {self.test_perf_string}')

        #cp_fp = format_checkpoint_filepath(self.cp_dir, bertcam='bert', epoch_number=ep)
        #save_model(self.model_name, self.model, cp_fp)


