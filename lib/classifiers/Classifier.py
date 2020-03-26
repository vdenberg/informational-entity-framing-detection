import time
from lib.utils import format_runtime, plot_scores
from lib.evaluate.StandardEval import my_eval

from sklearn.model_selection import learning_curve


class Classifier:
    """
    Generic Classifier that performs recurring machine learning tasks
    """
    def __init__(self, model, n_epochs, logger, patience, cp_dir, fig_dir, model_name, print_every):
        self.wrapper = model
        self.n_epochs = n_epochs
        self.logger = logger
        self.patience = patience
        self.fig_dir = fig_dir
        self.cp_dir = cp_dir
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
        self.cur_fold = ''

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

    def update_patience(self, val_f1):
        # if an improvement happens, we have full patience, if no improvement happens
        # patience goes down, if patience reaches zero, we stop training
        if val_f1 > self.prev_val_f1:
            self.current_patience = self.full_patience
        else:
            self.current_patience -= 1

    def unpack_fold(self, fold):
        self.cur_fold = fold['name']
        tr_bs, tr_lbs = fold['train_batches'], fold['train'].label
        dev_bs, dev_lbs = fold['dev_batches'], fold['dev'].label
        return tr_bs, tr_lbs, dev_bs, dev_lbs

    def validate_after_epoch(self, ep, elapsed, fold):
        tr_bs, tr_lbs, dev_bs, dev_lbs = self.unpack_fold(fold)

        tr_preds, tr_loss = self.wrapper.predict(tr_bs)
        val_preds, val_loss = self.wrapper.predict(dev_bs)

        epoch_name = f"{self.model_name}_fold{self.cur_fold}_ep{ep}"
        tr_mets, tr_perf = my_eval(tr_lbs, tr_preds, set_type='train', av_loss=tr_loss, name=epoch_name)
        val_mets, val_perf = my_eval(dev_lbs, val_preds, set_type='dev', av_loss=val_loss, name=epoch_name)

        self.logger.info(f" > Epoch{epoch_name} (took {elapsed}): {tr_perf}, {val_perf} (Best f1 so far: {self.best_val_f1})")

        self.wrapper.save_model(self.cp_dir, name=epoch_name)
        return tr_mets, tr_perf, val_mets, val_perf

    def train_all_epochs(self, fold):
        tr_bs, tr_lbs, dev_bs, dev_lbs = self.unpack_fold(fold)
        train_start = time.time()
        losses = []

        if self.model_name == 'BERT':
            elapsed = format_runtime(time.time() - train_start)
            tr_mets, tr_perf, val_mets, val_perf = self.validate_after_epoch(-1, elapsed, fold)

            if val_mets['f1'] > self.best_val_f1:
                self.best_val_f1 = val_mets['f1']

            losses.append((tr_mets['loss'], val_mets['loss']))

        for ep in range(self.n_epochs):
            self.wrapper.model.train()

            av_tr_loss, ep_elapsed = self.train_epoch(tr_bs)

            tr_mets, tr_perf, val_mets, val_perf = self.validate_epoch(tr_batches, tr_labels)

            if val_mets['f1'] > self.best_val_f1:
                self.best_val_f1 = val_mets['f1']

            losses.append((av_tr_loss, val_mets['loss']))

            if val_mets['f1'] > self.prev_val_f1:
                self.current_patience = self.full_patience
            else:
                self.current_patience -= 1

            self.update_patience(val_mets['f1'])

            self.prev_val_f1 = val_mets['f1']

            if not self.current_patience > 0:
                if val_mets['f1'] > 0.20:
                    self.logger.info(" Stopping training.")
                    break

        eps_elapsed = format_runtime(time.time() - train_start)
        return eps_elapsed, losses

    def test_model(self, fold):
        # test model
        preds, test_loss = self.wrapper.predict(fold['test_batches'])
        finalepochname =  f"{self.model_name}_fold{self.cur_fold}_finep{self.n_epochs}"
        test_metrics, test_perf_string = my_eval(fold['test'].label, preds,
                                                 name=finalepochname,
                                                 set_type='test', av_loss=test_loss)
        self.test_perf = [test_metrics['acc'], test_metrics['prec'], test_metrics['rec'], test_metrics['f1']]
        self.test_perf_string = test_perf_string

        self.logger.info(
            f' === DONE: Finished training {self.model_name} on Fold {fold["name"]} for {self.n_epochs} (took {self.train_time})')
        self.logger.info(f"{self.test_perf_string}")

        self.wrapper.save_model(self.cp_dir, name=finalepochname)

    def train_on_fold(self, fold):
        self.cur_fold = fold['name']
        train_elapsed, losses = self.train_all_epochs(fold['train_batches'],
                                                      fold['train'].label,
                                                      fold['dev_batches'],
                                                      fold['dev'].label)
        self.train_time = train_elapsed

        # plot learning curve
        loss_plt = plot_scores(losses)
        loss_plt.savefig(self.fig_dir + f'/{self.model_name}_trainval_loss.png', bbox_inches='tight')

        self.test_model(fold)





