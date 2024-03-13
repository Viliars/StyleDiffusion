from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # train
    train_batch_size = 16
    dataloader_num_workers = 1
    num_train_epochs = 100
    gradient_accumulation_steps = 4
    learning_rate = 1e-4
    lr_warmup_steps = 500
    #save_image_epochs = 1
    save_model_epochs = 10
    mixed_precision = "bf16"  # `no` for float32, `fp16` for automatic mixed precision
    lr_scheduler = "constant"

    output_dir = "latent-ffhq"  # the model name locally
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
