# Checkpoint Averaging

[Vaswani et. al. 2017](https://arxiv.org/abs/1706.03762) and [Izmailov et. al. 2018](https://arxiv.org/abs/1803.05407)
average the last N checkpoints during training, which leads to better generalisations and flatter optima.
[Wortsman et. al. 2022](https://arxiv.org/abs/2203.05482) propose model soups,
whereby weights of multiple fine-tuned models are averaged to produce a single model
that combines the strengths of the individual fine-tuned models.
We have also found checkpoint averaging to improve accuracy (~1-2% relative reduction in WER),
by averaging the best N checkpoints (from the same training run) based on validation WER.

To average the weights of a set of checkpoints, run the following command:

```bash
python caiman_asr_train/export/checkpoint_averaging.py --ckpts path/to/ckpt1.pt path/to/ckpt2.pt path/to/ckpt3.pt --output_path path/to/avg_ckpt.pt --model_config path/to/config.yaml
```

This script simply averages the weights of a list of checkpoints,
and does not find the best N checkpoints from a given training run -
this must be done manually at the moment, for example, by checking tensorboard logs.

```admonish
By default, CAIMAN-ASR already uses exponential moving average (EMA) weights,
which stabilise training and lead to smoother convergence by maintaining a running
average of model weights, with greater weighting on the most recent model weights.
However, checkpoint averaging can provide an improvement on top of EMA weights.
```
