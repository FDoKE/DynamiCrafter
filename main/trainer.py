import argparse, os, sys, datetime
from collections import OrderedDict
from typing import Dict, Any, Optional

import pytorch_lightning
from lightning_fabric.plugins import TorchCheckpointIO
from lightning_fabric.utilities.cloud_io import get_filesystem, _atomic_save
from lightning_fabric.utilities.types import _PATH
from omegaconf import OmegaConf
from torchtune.modules.peft import LoRALinear
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import torch
from typing_extensions import override

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utils import instantiate_from_config
from utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils_train import set_logger, init_workspace, load_checkpoints

class LoraCheckpointIo(TorchCheckpointIO):

    @override
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)

        while 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        state_dict_lora = OrderedDict()
        for layer_name in checkpoint.keys():
            if "lora_" in layer_name:
                state_dict_lora[layer_name] = checkpoint[layer_name]
        _atomic_save(state_dict_lora, path)



def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--seed", "-s", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--name", "-n", type=str, default="", help="experiment name, as saving folder")

    parser.add_argument("--base", "-b", nargs="*", metavar="base_config.yaml",
                        help="paths to base configs. Loaded from left-to-right. "
                             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
                        default=list())

    parser.add_argument("--train", "-t", action='store_true', default=False, help='train')
    parser.add_argument("--val", "-v", action='store_true', default=False, help='val')
    parser.add_argument("--test", action='store_true', default=False, help='test')

    parser.add_argument("--logdir", "-l", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--auto_resume", action='store_true', default=False, help="resume from full-info checkpoint")
    parser.add_argument("--auto_resume_weight_only", action='store_true', default=False,
                        help="resume from weight-only checkpoint")
    parser.add_argument("--debug", "-d", action='store_true', default=False, help="enable post-mortem debugging")

    return parser

def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))

def replace_linear_with_lora(module, rank, alpha, dropout, skip_layers, prefix=""):
    alphakoeff = rank * alpha
    for name, child in module.named_children():
        fullname = f"{prefix}.{name}" if prefix else name

        if not fullname.startswith("model") and not fullname.startswith("image_proj_model."):
            continue
        if isinstance(child, LoRALinear):
            continue

        if not all(sub not in fullname for sub in skip_layers):
            # you can skip another layers here if you want (temporal?)
            print ("SKIP LAYER: ", fullname)
            continue

        if isinstance(child, torch.nn.Linear):
            #print(fullname, child.weight.size())
            hasBias = child.bias is not None
            newLinear = LoRALinear(child.in_features, child.out_features, rank, alphakoeff, dropout, hasBias)
            newLinear.weight.data = child.weight.data.clone()
            if hasBias:
                newLinear.bias.data = child.bias.data.clone()
            module.__setattr__(name, newLinear)
        else:
            replace_linear_with_lora(child, rank, alpha, dropout, skip_layers, fullname)

def configure_lora(model, rank, alpha, dropout, skip_layers):
    print("LoRA configuration started. Rank: ", rank, ", alpha: ", alpha, ", dropout: ", dropout)

    for key, layer in model.named_modules():
        replace_linear_with_lora(layer, rank, alpha, dropout, skip_layers)

    for name, param in model.named_parameters():
        if ".lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    print("LoRA configured")

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))
    num_rank = int(os.environ.get('WORLD_SIZE'))

    parser = get_parser()
    ## Extends existing argparse by default Trainer attributes
    args, unknown = parser.parse_known_args()
    ## disable transformer warning
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)

    ## yaml configs: "model" | "data" | "lightning"
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    lora_config = lightning_config.get("lora", OmegaConf.create())

    ## setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir, config, lightning_config, global_rank)
    logger = set_logger(logfile=os.path.join(loginfo, 'log_%d:%s.txt' % (global_rank, now)))
    logger.info("@lightning version: %s [>=1.8 required]" % (pl.__version__))

    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Model *****")
    config.model.params.logdir = workdir
    model = instantiate_from_config(config.model)

    #apply lora layers to model

    if lora_config['enabled'] == True:
        rank = lora_config['rank']
        alpha = lora_config['alpha']
        dropout = lora_config['dropout']
        configure_lora(model, rank, alpha, dropout, lora_config['skip_layers'])

    ## load checkpoints
    model = load_checkpoints(model, config.model)

    ## register_schedule again to make ZTSNR work
    if model.rescale_betas_zero_snr:
        model.register_schedule(given_betas=model.given_betas, beta_schedule=model.beta_schedule, timesteps=model.timesteps,
                                linear_start=model.linear_start, linear_end=model.linear_end, cosine_s=model.cosine_s)

    ## update trainer config
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)

    #num_nodes = trainer_config.num_nodes
    #ngpu_per_node = trainer_config.devices
    #logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")

    ## setup learning rate
    base_lr = config.model.base_learning_rate
    bs = config.data.params.batch_size
    if getattr(config.model, 'scale_lr', True):
        model.learning_rate = num_rank * bs * base_lr
    else:
        model.learning_rate = base_lr


    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.datasets:
        logger.info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Trainer *****")
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"

    ## setup trainer args: pl-logger and callbacks
    train_logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
    train_logger = instantiate_from_config(train_logger_cfg)

    ## setup callbacks
    callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)
    callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    strategy_cfg = get_trainer_strategy(lightning_config)

    if lightning_config['optimize_matmul'] == True:
        torch.set_float32_matmul_precision('high')

    if lightning_config['compile'] == True:
        torch.compile(model)

    precision = lightning_config.get('precision', 32)
    accumulate_grad_batches = trainer_config['accumulate_grad_batches']
    benchmark = trainer_config['benchmark']
    max_steps = trainer_config['max_steps']
    log_every_n_steps = trainer_config['log_every_n_steps']
    val_check_interval = trainer_config['val_check_interval']
    gradient_clip_algorithm = trainer_config['gradient_clip_algorithm']
    gradient_clip_val = trainer_config['gradient_clip_val']
    accelerator = trainer_config['accelerator']

    # trainer_args = argparse.Namespace(**trainer_config)
    trainer = Trainer(precision=precision,
                      accumulate_grad_batches=accumulate_grad_batches,
                      max_steps=max_steps,
                      log_every_n_steps=log_every_n_steps,
                      val_check_interval=val_check_interval,
                      gradient_clip_algorithm=gradient_clip_algorithm,
                      gradient_clip_val=gradient_clip_val,
                      accelerator=accelerator,
                      benchmark=benchmark,
                      logger=train_logger,
                      callbacks=callbacks)

    if lora_config['save_only_lora'] == True:
        trainer.strategy.checkpoint_io = LoraCheckpointIo()

    ## allow checkpointing via USR1
    def melk(*args, **kwargs):
        ## run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    ## Running LOOP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Running the Loop *****")
    if args.train:
        try:
            if "strategy" in lightning_config and lightning_config['strategy'].startswith('deepspeed'):
                logger.info("<Training in DeepSpeed Mode>")
                ## deepspeed
                if precision == 16:
                    with torch.cuda.amp.autocast():
                        trainer.fit(model, data)
                else:
                    trainer.fit(model, data)
            else:
                logger.info("<Training in DDPSharded Mode>") ## this is default
                ## ddpsharded
                trainer.fit(model, data)
        except Exception:
            #melk()
            raise

    # if args.val:
    #     trainer.validate(model, data)
    # if args.test or not trainer.interrupted:
    #     trainer.test(model, data)