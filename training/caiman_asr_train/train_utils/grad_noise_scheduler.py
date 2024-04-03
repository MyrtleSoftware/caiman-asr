import torch

from caiman_asr_train.train_utils.distributed import print_once, unwrap_ddp


def switch_on_grad_noise_scheduler(cfg: dict, enc_freeze: bool) -> bool:
    """Return  boolean whether gradient noise will be used, according to the configuration.

    This function will return a boolean value defining whether gradient noise will
    be used during training. The boolean value depends on the configuration file. If
    the encoder is frozen, or the noise level is 0, then no gradient noise is used,
    and hence returns False.

    Parameters
    ----------
    cfg :
        The the model configuration from the yaml file.
    enc_freeze :
        Whether the encoder is frozen during training.

    Returns
    -------
        The return value, whether gradient noise will be used.
    """
    try:
        grad_noise_level = cfg["grad_noise_scheduler"]["noise_level"]
        if grad_noise_level <= 0.0 or enc_freeze:
            if enc_freeze:
                print_once(
                    "WARNING: Freezing the encoder turns off the encoder gradient "
                    "noise automatically."
                )
            return False
        else:
            print_once(
                "Noise will be added to the gradients of the encoder tensors during "
                "training."
            )
            return True
    except KeyError:
        print_once(
            "No gradient noise level information was found in the config file. Gradient "
            "noise scheduler will be None."
        )
        return False


class GradNoiseScheduler:
    """Scheduler that is used for calculation of the noise level depending on the step.

    This class initialises a scheduler that is used when gradient noise augmentation
    is applied to the training of the model. The noise is sampled from a gaussian
    distribution with mean=0 and standard deviation that decreases with time according to
    the equation: noise_level/(1 + step - start_step)**decay_const.
    The noise is added only to the gradients of the encoder tensors, and when the encoder
    is frozen, it is off by default.


    Parameters
    ----------
    seed
        The seed that will be used for the grad noise RNG
    noise_level
        The level of noise to be sampled
    decay_const
        Noise decay parameter
    start_step
        The step of the training that grad noise is switched on
    """

    def __init__(
        self,
        seed: int = 1,
        noise_level: float = 0.15,
        decay_const: float = 0.55,
        start_step: int = 1,
    ):
        assert (
            noise_level > 0
        ), f"noise level noise_level has to be > 0, value given: {noise_level}."
        assert (
            decay_const >= 0
        ), f"decay_const exponent has to be >= 0, value given: {decay_const}."
        assert start_step >= 1, f"start step has to be >= 1, value given: {start_step}."
        self.noise_level = noise_level
        self.decay_const = decay_const
        self.start_step = start_step
        self.rng_grad_noise = torch.Generator(device="cuda")
        self.rng_grad_noise.manual_seed(seed)

    def std(self, step: int) -> float:
        """Calculate the noise level based on the current step."""
        return self.noise_level / (1 + step - self.start_step) ** self.decay_const

    def switch_on(self, step: int) -> bool:
        """Decide whether noise will be added based on step the training is
        currently in, and the start_step when the noise is switched on.
        """
        return step >= self.start_step

    def add_grad_noise(self, model, step: int, world_size: int):
        """
        Update the gradients with noise sampled from a gaussian distribution according
        to the step.
        """

        if self.switch_on(step):
            model_encoder = unwrap_ddp(model).encoder
            for param_name, param in model_encoder.named_parameters():
                noise = torch.normal(
                    mean=0.0,
                    std=self.std(step),
                    size=param.size(),
                    device=param.device,
                    dtype=param.dtype,
                    generator=self.rng_grad_noise,
                )
                # each GPU adds noise to the tensors grads, but it is only needed once
                noise.data.div(world_size)
                param.grad.data.add_(noise)
        return model
