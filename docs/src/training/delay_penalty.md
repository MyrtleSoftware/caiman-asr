# Delay Penalty

Delay penalty is an algorithm that was introduced in [this paper](https://arxiv.org/abs/2211.00490) to reduce emission latency of RNN-T models. The idea is to penalize delayed emission of non-blank tokens according to the specific frame indexes during training. The algorithm was implemented directly into the Apex loss calculation by introducing an auxiliary term to the vanilla RNN-T loss. The implementation supports two options of setting the delay penalty: a constant value or a stepwise scheduler.

It is important to note that reducing emission latency may result in a slight degradation of WER. In general, the higher the delay penalty, the lower the emission latency and the higher the WER.

## Constant delay penalty

The original paper describes fixing the delay penalty at a constant value `lambda` throughout the training. This setup can be used by passing the following argument to the training script:

```bash
./scripts/train.sh --delay_penalty=lambda
```

The paper shows that the optimal values of `lambda` are in range of [0.001, 0.01].

## Stepwise

Our experiments have shown that a non-constant delay penalty scheduler achieves a better trade-off between WER and emission latency. Enabling delay penalty before the model has learned a small amount of language was empirically found to prevent the RNN model from converging below a WER of 100%. However, this requires having a good estimate __before training__ of the point at which the model goes below 100% WER. The stepwise schedule avoids this by stepping from an initial delay penalty to a final delay penalty at a WER threshold. The stepwise schedule can be enabled by passing:

```bash
--delay_penalty 'wer_schedule' \
--dp_initial_value <value before toggle step> \
--dp_final_value <value after toggle step> \
--dp_toggle_step <fallback training step to toggle on> \
--dp_wer_threshold <WER threshold to trigger step>
```

The default values for these arguments were the values used to train the v1.13.0 models.

### Next steps

To evaluate the emission latency of your model, see the [Emission Latency documentation](./emission_latency.md).
