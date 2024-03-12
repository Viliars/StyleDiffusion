from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 5
    mixed_precision = "bf16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "latent-ffhq"  # the model name locally

    overwrite_output_dir = True  # overwrite the old model
    seed = 0


config = TrainingConfig()
