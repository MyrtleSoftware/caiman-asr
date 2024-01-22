# CustomLSTM

We can swap out the standard PyTorch LSTM for one of our own [CustomLSTM](./../lib/src/rnnt_ext/custom_lstm/) by changing the
config file in directory [configs](../configs/) to read:

```
custom_lstm: true
```

This is the default as this custom implementation is required for Random State Passing (RSP).

The CustomLSTM exposes the internal LSTM machinery to facilitate experimentation. For example, encoder recurrent
weight dropout (as used in [Zeyer et al., 2021](https://arxiv.org/pdf/2104.03006v2.pdf))
can be switched on in the config file using:

```
custom_lstm: true
enc_rw_dropout: 0.1
```

We have two implementations of our Custom LSTM, one written in CUDA that matches the PyTorch LSTM's speed and a second TorchScript version that runs ~ 3.5x slower.

On a server with eight A100s, the PyTorch LSTM had a throughput of 440 utterances/second while training the base model.
The CUDA Custom LSTM had a throughput of 375 utterances/second.

## Quantization

Inference can be performed with quantized tensors in all the LSTM layers.

As there is no PyTorch support for quantization of LSTM layers,
quantization can only be used with the CustomLSTM. This can be achieved
by changing the config file in directory [configs](../configs/) to read

```
custom_lstm: true
quantize: true
```

The CUDA version is used by default but does not support quantization; hence, we fallback to the TorchScript version when quantization is requested.

The values of the tensors are quantized to
[BrainFloat16](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)
and for tensor multiplication, the tensors are further quantized into
Block Floating Point 16 with blocks of size 8.
The quantization classes are defined in [quantize.py](./../rnnt_train/common/quantize.py),
and include the [QPyTorch package](https://github.com/Tiiiger/QPyTorch).
