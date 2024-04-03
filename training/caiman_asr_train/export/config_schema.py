from pydantic import BaseModel


class Schema(BaseModel):
    """
    RNN-T inference schema base class.

    These classes ingest the yaml schemas in configs/ dropping any fields that are not
    required at inference time.

    Any training-specific config arguments that don't affect inference should be placed
    in _allow_ignore.
    """

    _allow_ignore = set()

    class Config:
        extra = "forbid"

    def __init__(self, **kwargs):
        keys = set(kwargs.keys())
        for k in self._allow_ignore:
            if k in keys:
                kwargs.pop(k)
        super().__init__(**kwargs)


class FilterbankFeaturesSchema(Schema):
    dither: float
    n_fft: int
    n_filt: int
    normalize: str
    sample_rate: int
    window: str
    window_size: float
    window_stride: float

    _allow_ignore = {
        "stats_path",
    }


class FrameSplicingSchema(Schema):
    frame_stacking: int
    frame_subsampling: int


class ModelSchema(Schema):
    enc_n_hid: int
    enc_post_rnn_layers: int
    enc_pre_rnn_layers: int
    enc_stack_time_factor: int
    in_feats: int
    joint_n_hid: int
    pred_n_hid: int
    pred_rnn_layers: int

    _allow_ignore = {
        "custom_lstm",
        "enc_batch_norm",
        "enc_dropout",
        "enc_freeze",
        "enc_rw_dropout",
        "forget_gate_bias",
        "hidden_hidden_bias_scale",
        "weights_init_scale",
        "joint_apex_relu_dropout",
        "joint_apex_transducer",
        "joint_dropout",
        "pred_batch_norm",
        "pred_dropout",
        "pred_rw_dropout",
        "quantize",
        "gpu_unavailable",
        "hard_activation_functions",
        "enc_lr_factor",
        "pred_lr_factor",
        "joint_enc_lr_factor",
        "joint_pred_lr_factor",
        "joint_net_lr_factor",
    }


class TokenizerSchema(Schema):
    labels: list
    sentpiece_model: str

    _allow_ignore = {
        "sampling",
    }


class InputValSchema(Schema):
    filterbank_features: FilterbankFeaturesSchema
    frame_splicing: FrameSplicingSchema

    _allow_ignore = {
        "audio_dataset",
    }


class RNNTInferenceConfigSchema(Schema):
    input_val: InputValSchema
    rnnt: ModelSchema
    tokenizer: TokenizerSchema

    _allow_ignore = {
        "input_train",
        "grad_noise_scheduler",
    }
