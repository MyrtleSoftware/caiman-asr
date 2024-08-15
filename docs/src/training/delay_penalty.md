# Delay Penalty

Delay penalty is an algorithm that was introduced in [this paper](https://arxiv.org/abs/2211.00490) to reduce emission latency of RNN-T models. The idea is to penalize delayed emission of non-blank tokens according to the specific frame indexes during training. The algorithm was implemented directly into the Apex loss calculation by introducing an auxilliary term to the vanilla RNN-T loss. The implementation supports two options of setting the delay penalty: a constant value or a linear scheduler.

It is important to note that reducing emission latency can come at the cost of small WER degradation. In general, the higher the delay penalty, the lower the emission latency and the higher the WER.

## Constant delay penalty

The original paper describes fixing the delay penalty at a constant value `lambda` throughout the training. This setup can be used by passing the following argument to the training script:

```admonish
`--delay_penalty=lambda`
```

The paper shows that the optimal values of `lambda` are in range of \[0.001, 0.01\].

## Delay penalty with a scheduler

Our experiments have shown that a linear delay penalty scheduler achieves a better trade-off between the WER and emission latency. The scheduler looks as follows. At first, the model is trained for several thousands of steps without any penalty. In the next step, the penalty is increased step-wise to a relatively large value. Finally, the penalty is increased linearly until it reaches its final value after which it is kept constant. This setup can be used by passing the following arguments to the training script:

```bash
--delay_penalty 'linear_schedule' \
--dp_warmup_steps <number of warmup steps> \
--dp_warmup_penalty <penalty during a warmpup period> \
--dp_ramp_penalty <penalty value to ramp to at step = (--dp_warmup_steps + 1)> \
--dp_final_steps <final step number until which the penalty keeps increasing linearly> \
--dp_final_penalty <final penalty value past step = (--dp_final_steps)>
```

### Default linear schedule

The following setup was used to train the models for release v1.12.0,
and is on by default:

```bash
--delay_penalty 'linear_schedule' \
--dp_warmup_steps 5000 \
--dp_warmup_penalty 0.0 \
--dp_ramp_penalty 0.007 \
--dp_final_steps 20000 \
--dp_final_penalty 0.01
```
