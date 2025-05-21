import numpy as np
import math
from lightning.pytorch.callbacks import Callback
import torch
import torch.nn as nn

class StochasticRACLoRA(Callback):
    """
    - Every `merge_frequency` epochs, merges LoRA weights into the base model,
      and then does a coin-flip to pick which matrix (A or B) will be trained
      in the *next* epoch.
    - Depending on the flags:
      * init_train_zero: if True, re-init the chosen (train) matrix to zeros
        at merge time; if False, keep its old values
      * gaussian_resample: if True, re-init the frozen matrix with Gaussian
        at merge time; if False, keep its old values
    """

    def __init__(
        self,
        merge_frequency=1,
        random_training=True,
        prob=0.5,
        deterministic_init=True,
        init_train_zero=False,
        gaussian_resample=False,
        seed=42,
):
        """
        Parameters
        ----------
        merge_frequency : int
            Merge + partial re-init every N epochs
        random_training : bool
            If False, do nothing special, preserving old code.
        prob : float
            Probability of training A (vs. B)
        init_train_zero : bool
            If True, the matrix chosen for training is re-init with zeros
            at merge time; if False, keep its old values.
        gaussian_resample : bool
            If True, the matrix that is *not* chosen for training is re-init
            with Gaussian at merge time; if False, keep its old values.
        seed : int
            Seed for reproducibility of coin flips.
        """
        super().__init__()
        self.merge_frequency = merge_frequency
        self.random_training = random_training
        self.prob = prob
        self.deterministic_init = deterministic_init
        self.init_train_zero = init_train_zero
        self.gaussian_resample = gaussian_resample
        self.rs = np.random.RandomState(seed)

        # We'll store who is training for the *next* epoch.
        self.next_train_is_A = None

    # def on_fit_start(self, trainer, pl_module):
    #     """
    #     Runs once before the first epoch starts:
    #       - We do an initial coin-flip to decide which matrix to train at epoch=0.
    #       - We do *not* do partial re-init here, because no merge has happened yet.
    #     """
    #     if not self.random_training:
    #         return  # no-op

    #     c = self.rs.binomial(n=1, p=self.prob, size=1)[0]
    #     self.next_train_is_A = (c == 0)
    #     if self.next_train_is_A:
    #         print("[StochasticRACLoRA] Initial -> train A, freeze B.")
    #         pl_module.config.lora_train_A = True
    #         pl_module.config.lora_train_B = False
    #     else:
    #         print("[StochasticRACLoRA] Initial -> train B, freeze A.")
    #         pl_module.config.lora_train_A = False
    #         pl_module.config.lora_train_B = True

    #     pl_module._freeze_non_lora_weights()

    def on_fit_start(self, trainer, pl_module):
        """
        Runs once before the first epoch starts:
        - We flip a coin to decide which matrix is trained at epoch=0.
        - We set init_method_A/B so that the trained matrix is zero-inited,
            and the frozen matrix is gaussian-inited.
        - We call reset_lora_parameters() to do that reinit,
            then freeze the matrix not being trained.
        """
        if self.deterministic_init:
            return
        
        if not self.random_training:
            return  # no-op

        # Flip coin (Bernoulli) to decide which matrix to train
        c = self.rs.binomial(n=1, p=self.prob, size=1)[0]

        if c == 0:
            # Train A, freeze B
            pl_module.config.lora_train_A = True
            pl_module.config.lora_train_B = False
            # A -> zero init, B -> gaussian
            pl_module.config.lora_init_method_A = "zero"
            pl_module.config.lora_init_method_B = "gaussian"
            msg = "[StochasticRACLoRA] Initial -> train A, freeze B."
        else:
            # Train B, freeze A
            pl_module.config.lora_train_A = False
            pl_module.config.lora_train_B = True
            # B -> zero init, A -> gaussian
            pl_module.config.lora_init_method_A = "gaussian"
            pl_module.config.lora_init_method_B = "zero"
            msg = "[StochasticRACLoRA] Initial -> train B, freeze A."

        print(msg)

        # Now re-init LoRA parameters with the new init methods
        pl_module.reset_lora_parameters()

        # Freeze whichever side is not being trained
        pl_module._freeze_non_lora_weights()



    def on_train_epoch_end(self, trainer, pl_module):
        """
        After each epoch:
          1. If (epoch+1) is a multiple of `merge_frequency`, we:
             - Merge LoRA weights,
             - Decide who trains next epoch (coin-flip),
             - Partially re-init (train side vs. frozen side),
             - Flip flags & freeze accordingly.
          2. Otherwise, do nothing special.
        """
        if not self.random_training:
            return  # no-op

        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.merge_frequency == 0:
            # 1) Merge
            pl_module.merge_lora_weights()

            # 2) Flip coin for the *next* epoch
            c = self.rs.binomial(n=1, p=self.prob, size=1)[0]
            next_train_is_A = (c == 0)
            if next_train_is_A:
                print(f"[StochasticRACLoRA] Next epoch -> train A, freeze B.")
            else:
                print(f"[StochasticRACLoRA] Next epoch -> train B, freeze A.")

            # 3) Partial re-init
            self.partial_reinit(pl_module, train_A=next_train_is_A)

            # 4) Update flags
            pl_module.config.lora_train_A = next_train_is_A
            pl_module.config.lora_train_B = not next_train_is_A
            pl_module._freeze_non_lora_weights()

            self.next_train_is_A = next_train_is_A

    def partial_reinit(self, pl_module, train_A):
        """
        Partially re-initialize LoRA parameters:
          - If train_A=True, then A is about to be trained, B is frozen
            => if init_train_zero=True, re-init A with zeros, else keep old
            => if gaussian_resample=True, re-init B with Gaussian, else keep old
          - If train_A=False, then B is about to be trained, A is frozen
            => if init_train_zero=True, re-init B with zeros, else keep old
            => if gaussian_resample=True, re-init A with Gaussian, else keep old
        """
        # We have 3 layers: l1_lora_A, l1_lora_B, l2_lora_A, ...
        # We'll handle them in a loop or individually.

        # Helper function: zero init
        def zero_init(tensor):
            with torch.no_grad():
                tensor.zero_()

        # Helper function: gaussian init
        def gaussian_init(tensor):
            with torch.no_grad():
                nn.init.normal_(tensor, mean=0.0, std=1.0)

        # We'll need to check the named parameters or direct attribute references
        # since we are in LitMNISTLoRA class.
        # Direct references could be:
        #   pl_module.l1_lora_A, pl_module.l1_lora_B, etc.
        # But let's do a small helper that picks (A or B) from each layer:
        # Layer naming is consistent: l1_lora_A, l1_lora_B, l2_lora_A, ...
        # We'll do something like:

        for layer_idx in [1, 2, 3]:
            # get A and B
            A_name = f"l{layer_idx}_lora_A"
            B_name = f"l{layer_idx}_lora_B"
            A_param = getattr(pl_module, A_name)
            B_param = getattr(pl_module, B_name)

            if train_A:
                # We are training A => freeze B
                if self.init_train_zero:
                    # Re-init A with zero
                    zero_init(A_param)
                # else keep old A

                if self.gaussian_resample:
                    # Re-init B with gaussian
                    gaussian_init(B_param)
                # else keep old B
            else:
                # We are training B => freeze A
                if self.init_train_zero:
                    # Re-init B with zero
                    zero_init(B_param)

                if self.gaussian_resample:
                    # Re-init A with gaussian
                    gaussian_init(A_param)
