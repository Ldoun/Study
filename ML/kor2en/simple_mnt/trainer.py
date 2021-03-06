from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils
from  torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from simple_mnt.utils import get_grad_norm,get_parameter_norm
from ignite.contrib.handlers.tensorboard_logger import *

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class MaximumLikelihoodEstimationEngine(Engine):
    def __init__(self, func, model, crit, optimizer,lr_scheduler, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config

        super().__init__(func)

        self.best_loss =np.inf
        self.best_model = None
        self.scaler = GradScaler()


    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()
        if engine.state.iteration % engine.config.iteration_per_update == 1 or engine.config.iteration_per_update == 1:
            engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device),mini_batch.src[1]) #tensor,length
        mini_batch.tgt = (mini_batch.tgt[0].to(device),mini_batch.tgt[1])
        
        x,y = mini_batch.src,mini_batch.tgt[0][:,1:]  #<BOS> 제외 정답문장 1번 단어부터 비교
        #|x| = (batch_size,length)
        #|y| = (batch_size,length)
        print(x[0].size())
        print(y.size())


        with autocast():     
            y_hat = engine.model(x,mini_batch.tgt[0][:,:-1])
            #|y_hat| = (batch_size,length,ouput_size)
            loss = engine.crit(
                y_hat.contiguous().view(-1,y_hat.size(-1)),
                y.contiguous().view(-1)
            )

            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)
            loss.backward()

        if engine.config.gpu_id >=0:
            engine.scaler.scale(backward_target).backward(retain_graph=True)
        else:
            backward_target.backward()
        
        word_count = int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters())) #모델의 복잡도 학습됨에 따라 커져야함
        g_norm = float(get_grad_norm(engine.model.parameters()))    #클수록 뭔가 배우는게 변하는게 많다 (학습의 안정성)

        if engine.state.iteration %     engine.config.iteration_per_update == 0:
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_gr_norm,
                #norm_type=2,
            )

            if engine.config.gpu_id >=0:
                engine.scaler.step(engine.optimizer)
                engine.scaler.update()
            else:
                engine.optimizer.step()

            if engine.config.use_noam_decay and engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

        loss = float(loss/word_count)
        ppl = np.exp(loss)   

        return {
            'loss': loss,
            'ppl': ppl,
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
            '|p_param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device 
            mini_batch.src = (mini_batch.src[0].to(device),mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device),mini_batch.tgt[1])
            
            x, y = mini_batch.src, mini_batch.tgt[:,1:]
            print(x.size())
            #|x| = (batch_size,length)
            #|y| = (batch_size,length)
            
            with autocast():
                y_hat = engine.model(x,mini_batch.tgt[0][:,:-1])
                #|y_hat| = (batch_size,n_class)
                
                loss = engine.crit(
                    y_hat.contiguous().view(-1,y_hat.size(-1)),
                    y.contiguous().view(-1),
                )
            
        word_count = int(mini_batch.tgt[1].sum())
        loss = float(loss/word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl
        }

    @staticmethod
    def test(engine,mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device 
            mini_batch.src = (mini_batch.src[0].to(device),mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device),mini_batch.tgt[1])

            x, y = mini_batch.src, mini_batch.tgt[:,1:]

            with autocast():
                y_hat = engine.model(x,mini_batch.tgt[0][:,:-1])
                #|y_hat| = (batch_size,n_class)
                
                loss = engine.crit(
                    y_hat.contiguous().view(-1,y_hat.size(-1)),
                    y.contiguous().view(-1),
                )

        word_count = int(mini_batch.tgt[1].sum())
        loss = float(loss/word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl
        }

    @staticmethod
    def attach(train_engine, validation_engine, verbose = VERBOSE_BATCH_WISE): 
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform = lambda x: x[metric_name]).attach(
                engine,
                metric_name
            )

        training_metric_name = ['loss', 'ppl', '|p_param|', '|g_param|']

        for metric_name in training_metric_name:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format = None, ncols=120)
            pbar.attach(train_engine, training_metric_name)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |p_params| = {:.2e} |g_param| = {:.2e} loss = {:.4e} ppl = {:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|p_param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['ppl'],
                ))

        validation_metrics_name = ['loss','ppl']

        for metrics in validation_metrics_name:
            attach_running_average(validation_engine, metrics)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format = None, ncols=120)
            pbar.attach(validation_engine, validation_metrics_name)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss = {:.4e} ppl = {:.4f} best_loss = {:.4e}'.format(             
                    engine.state.metrics['loss'],
                    engine.state.metrics['ppl'],
                    engine.best_loss
                ))

    @staticmethod
    def resume_training(engine,resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)
        
    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())
        
    @staticmethod
    def save_model(engine, train_engine, config,src_vocab,tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        model_fn =  config.model_fn.split('.')
        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (
                                        avg_train_loss,
                                        np.exp(avg_train_loss)
                                        ),
                                    '%.2f-%.2f' % (
                                        avg_valid_loss,
                                        np.exp(avg_valid_loss)
                                    )] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        torch.save(
            {
                'model':engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab' : src_vocab,
                'tgt_vocab' : tgt_vocab,
            },model_fn
            )




class SingleTrainer():
    def __init__(self,target_engine_class,config):
        self.target_engine_class = target_engine_class
        self.config = config
        self.tb_logger = TensorboardLogger(log_dir = config.log_dir)
        super().__init__()

    def train(
        self,
        model,crit,optimizer,train_loader,valid_loader,
        src_vocab,tgt_vocab,
        n_epochs,
        lr_scheduler = None
    ):
        self.train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model, crit, optimizer,lr_scheduler, self.config
        )

        self.valid_engine = self.target_engine_class(
            self.target_engine_class.validate,
            model, crit, optimizer=None,lr_scheduler = None,
            config = self.config
        )

        self.tb_logger.attach_output_handler(
            self.train_engine,
            event_name =Events.EPOCH_COMPLETED,
            tag="training",
            metric_names = "all"
        )

        self.tb_logger.attach_output_handler(
            self.valid_engine,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names = "all",
            global_step_transform=global_step_from_engine(self.train_engine)
        )

        self.target_engine_class.attach(
            self.train_engine,
            self.valid_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, valid_engine, valid_loader):
            valid_engine.run(valid_loader, max_epochs =1)

        self.train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #event
            run_validation, #func
            self.valid_engine, valid_loader #args
        )

        self.valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #event
            self.target_engine_class.check_best #func
        )

        self.train_engine.add_event_handler(
            Events.STARTED,
            self.target_engine_class.resume_training,
            self.config.init_epoch,
        )

        self.valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #event
            self.target_engine_class.save_model, # func
            self.train_engine, self.config,
            src_vocab,tgt_vocab #args
        )

        self.train_engine.run(
            train_loader,
            max_epochs = self.config.n_epochs
        )

        return model

    def test(self,test_loader):
        print('--------------train-------------------')
        self.valid_engine.run(
            test_loader,
            max_epochs=1
        )