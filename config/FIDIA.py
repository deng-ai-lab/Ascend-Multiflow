import ml_collections

def get_config(name):
    return globals()[name]()

def base():
    config = ml_collections.ConfigDict()
    config.sample = sample = ml_collections.ConfigDict()
    config.ds = ds = ml_collections.ConfigDict()
    config.train = train = ml_collections.ConfigDict()
    config.ablation = ablation = ml_collections.ConfigDict()

    # *------default--------------
    config.task="" #motif; ligand
    config.model_type=""
    config.checkpoint_path=""
    config.load_esmf = True
    config.seed = 42
    config.mixed_precision = "no"
    config.allow_tf32=False
    # *----------------------------EXPERIMENT LOG--------------------------------------
    config.EXPTAG = "default"
    config.save_dir = 'data/ckpt'
    config.log_dir = 'logs'
    config.wandb = True
    config.run_name_wandb = "default"
    
    # *--------------- dataset ----------------------------------------
    ds.dataset_type=""
    ds.eval_size = 64
    ds.train_batch_size = 32
    ds.eval_batch_size = 8
    ds.test_batch_size = 15
    
    # *--------------EVAL -------------------------------------------------------
    config.disable_eval = False
    config.disable_first_eval = False
    config.eval_mode="cond"
    config.eval_bmk = False
    
    # * --------------- sample ----------------------------------------
    sample.sampling_temp = 1.0
    sample.sampling_temp_eval = 0.1
    sample.num_seq_per_target = 16
    sample.num_seq_per_target4eval = 8
  
    # *---------------train---------------------------------
    train.mode="" #IAPO; GRPO
    train.num_samples_per_inner_epoch = 0
    train.adv_clip_max=5
    train.clip_range = 0.3
    train.max_grad_norm = 1.0
    train.gradient_accumulation_steps = 8
    train.num_inner_epochs = 1
    train.fix_train_motif=False
    train.adv_normalize_method="zscore"
    # *---------------optimize------------------------
    train.only_encoder=False
    train.freeze_mpnn=False
    train.beta = 0.0
    train.reward_design="Composite"
    train.use_8bit_adam = False
    train.learning_rate = 5e-5
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.total_epochs = 10
    train.scheduler = True

    # *---ablation--
    ablation.mode=""
    ablation.grmsd_not_plateau=False
  
    return config

def IAPO():
    config = base()
    config.train.mode = "IAPO"
    config.sample.sampling_temp = 0.3
    return config

def GRPO():
    config = base()
    config.train.mode = "GRPO"
    config.sample.sampling_temp = 0.3
    return config

def EVAL():
    config = base()
    config.wandb=False
    config.eval_bmk=True
    config.train.mode="EVAL"
    return config

def debug():
    config = base()
    config.EXPTAG = "DEBUG"
    config.train.mode = "IAPO"
    config.sample.sampling_temp = 0.3
    config.disable_eval = True
    config.ds.train_batch_size = 4
    config.train.gradient_accumulation_steps = 2
    config.sample.num_seq_per_target = 4
    config.sample.num_seq_per_target4eval = 2
    config.load_esmf = False
    return config
