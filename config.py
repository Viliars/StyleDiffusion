from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # train
    train_batch_size = 64
    dataloader_num_workers = 4
    num_train_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 2.0e-05
    lr_warmup_steps = 500
    #save_image_epochs = 1
    save_model_epochs = 5
    mixed_precision = "bf16"  # `no` for float32, `fp16` for automatic mixed precision
    lr_scheduler = "constant"
    use_ema = True

    output_dir = "latent-ffhq-256"  # the model name locally
    overwrite_output_dir = True  # overwrite the old model
    seed = 0

    # logging
    report_to = "tensorboard"
    logging_dir = "logs"

    # sheduler
    ddpm_num_steps = 1000
    ddpm_beta_schedule = "linear"
    prediction_type = "epsilon"

    # optimization
    enable_xformers_memory_efficient_attention = True
    
    # hz
    max_grad_norm = 1.0


config = TrainingConfig()
