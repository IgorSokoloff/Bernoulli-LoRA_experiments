import os
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
#from lightning.pytorch.utilities.seed import seed_everything
from lightning import seed_everything
import numpy as np

from callbacks import LoRAMergeCallback
from callbacks_stochastic import StochasticRACLoRA  # new callback

from config import Config
from models.lit_mnist import LitMNIST
from models.lit_mnist_lora import LitMNISTLoRA

def train_model(model, config, is_lora_train,
                use_stochastic=False,
                prob=0.5,
                deterministic_init=False,
                init_train_zero=False,
                gaussian_resample=False,
                seed=42
                ):
    callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=40)]

    if is_lora_train:
        if use_stochastic:
            # Use the new partial re-init approach
            callbacks.append(StochasticRACLoRA(
                merge_frequency=config.merge_frequency,
                random_training=True,
                prob=prob,
                deterministic_init=deterministic_init,
                init_train_zero=init_train_zero,
                gaussian_resample=gaussian_resample,
                seed=seed
            ))
        else:
            # old approach with LoRAMergeCallback
            if config.merge_frequency > 0:
                callbacks.append(LoRAMergeCallback(config.merge_frequency))

    trainer = L.Trainer(
        accelerator="cuda",
        devices=[0],
        max_epochs=config.max_epochs,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=callbacks,
        deterministic=True
    )
    trainer.fit(model)
    trainer.test()
    return trainer


def run_lora_experiment(
    rank,
    train_A=True,
    train_B=True,
    init_method_A='kaiming',
    init_method_B='zero',
    merge_frequency=1,
    use_stochastic=False,
    prob=0.5,
    deterministic_init=True,
    init_train_zero=False,
    gaussian_resample=False,
    seed=42
):
    config = Config(
        lora_rank=rank,
        lora_train_A=train_A,
        lora_train_B=train_B,
        lora_init_method_A=init_method_A,
        lora_init_method_B=init_method_B,
        merge_frequency=merge_frequency
    )
    state_dict = torch.load("base_model.pt")
    model = LitMNISTLoRA(config)
    model.load_state_dict(state_dict, strict=False)
    model.class_names = [5, 6, 7, 8, 9]
    model.min_class = 5

    trainer = train_model(
        model,
        config,
        is_lora_train=True,
        use_stochastic=use_stochastic,
        prob=prob,
        deterministic_init=deterministic_init,
        init_train_zero=init_train_zero,
        gaussian_resample=gaussian_resample,
        seed=seed
    )
    return trainer.test(model)[0]['test_acc']



def run_lora_experiment_multiple_seeds(
    seeds,
    rank=1,
    train_A=True,
    train_B=True,
    init_method_A='gaussian',
    init_method_B='zero',
    merge_frequency=0,
    use_stochastic=False,
    prob=0.5,
    deterministic_init=True,
    init_train_zero=False,
    gaussian_resample=False,
):
    """
    Runs the LoRA experiment multiple times (each with a different seed) and
    returns various accuracy statistics.
    """
    
    print("Parameters passed to run_lora_experiment_multiple_seeds:")
    for param, value in locals().items():
        print(f"{param}: {value}")
    
    accuracies = np.zeros(len(seeds))
    for i,seed in enumerate(seeds):
        # Set the global random seed for PyTorch, Numpy, etc.
        seed_everything(seed, workers=True)
        acc = run_lora_experiment(
            rank=rank,
            train_A=train_A,
            train_B=train_B,
            init_method_A=init_method_A,
            init_method_B=init_method_B,
            merge_frequency=merge_frequency,
            use_stochastic=use_stochastic,
            prob=prob,
            deterministic_init=deterministic_init,
            init_train_zero=init_train_zero,
            gaussian_resample=gaussian_resample,
            seed=seed
        )
        print(f"[Seed={seed}] Test Accuracy: {acc:.4f}")
        accuracies[i] = acc

    median_acc = np.median(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print(f"Accuracies = {accuracies}")
    print(f"Mean accuracy = {mean_acc:.4f}")
    print(f"Median accuracy = {median_acc:.4f}")
    print(f"Standard deviation = {std_acc:.4f}")
    return accuracies, median_acc, mean_acc, std_acc


def run_baseline_experiment():
    """Run baseline MNIST model training experiment.
    
    Returns
    -------
    float
        Test accuracy of the baseline model
        
    Notes
    -----
    - Checks for existing model checkpoint to avoid retraining
    - If no checkpoint exists, trains new model and saves checkpoint
    - Uses default Config parameters for baseline training
    """
    model_filepath = "base_model.ckpt"
    if os.path.isfile(model_filepath):
        print("Found existing base model checkpoint. Skipping training of the base model")
        trainer = L.Trainer(
            accelerator="cuda",
            devices=[0],
            logger=CSVLogger(save_dir="logs/"),
        )
        model = LitMNIST.load_from_checkpoint(model_filepath, config=Config())
    else:
        config = Config()
        model = LitMNIST(config)
        trainer = train_model(model, config, False)
        trainer.save_checkpoint("base_model.ckpt")
        torch.save(model.state_dict(), 'base_model.pt')
    return trainer.test(model)[0]['test_acc']




#------------------------------------

#old
# def train_model(model, config, is_lora_train, use_stochastic=False, prob=0.5):
#     """
#     If use_stochastic=True, attach StochasticRACLoRA so that
#     each epoch merges, resets, and picks A or B at random.
    
#     Otherwise, if config.merge_frequency>0, use old LoRAMergeCallback.
#     """
#     callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=10)]

#     if is_lora_train:
#         if use_stochastic:
#             # Use the new coinflip-based approach
#             callbacks.append(StochasticRACLoRA(random_training=True,
#                                                prob=prob,
#                                                seed=42))
#         else:
#             # The old approach (merge every 'merge_frequency' epochs)
#             if config.merge_frequency > 0:
#                 callbacks.append(LoRAMergeCallback(config.merge_frequency))

#     trainer = L.Trainer(
#         accelerator="cuda",
#         devices=[0],
#         max_epochs=config.max_epochs,
#         logger=CSVLogger(save_dir="logs/"),
#         callbacks=callbacks,
#         deterministic=True
#     )
#     trainer.fit(model)
#     trainer.test()
#     return trainer


#old
# def run_lora_experiment(
#     rank,
#     train_A=True,
#     train_B=True,
#     init_method_A='kaiming',
#     init_method_B='zero',
#     merge_frequency=1,
#     use_stochastic=False,  # <--- new param
#     prob=0.5               # <--- new param
# ):

#     # Build config
#     config = Config(
#         lora_rank=rank,
#         lora_train_A=train_A,
#         lora_train_B=train_B,
#         lora_init_method_A=init_method_A,
#         lora_init_method_B=init_method_B,
#         merge_frequency=merge_frequency
#     )
#     # Load baseline state dict
#     state_dict = torch.load("base_model.pt")

#     # Create LoRA model
#     model = LitMNISTLoRA(config)
#     model.load_state_dict(state_dict, strict=False)

#     # Now adapt to classes [5..9] (like in the original code)
#     model.class_names = [5, 6, 7, 8, 9]
#     model.min_class = min(model.class_names)

#     # Train using our updated train_model function
#     trainer = train_model(model, config,
#                           is_lora_train=True,
#                           use_stochastic=use_stochastic,
#                           prob=prob)
#     return trainer.test(model)[0]['test_acc']
