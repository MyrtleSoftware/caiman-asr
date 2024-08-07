# TensorBoard

The training scripts write TensorBoard logs to /results during training.

To monitor training using TensorBoard, launch the port-forwarding TensorBoard container in another terminal:

```bash
./scripts/docker/launch_tb.sh <RESULTS> <OPTIONAL PORT NUMBER> <OPTIONAL NUM_SAMPLES>
```

If `<OPTIONAL PORT NUMBER>` isn't passed then it defaults to port 6010.
`NUM_SAMPLES` is the number of steps that TensorBoard will sample from the log and plot.
It defaults to 1000.

Then navigate to `http://traininghostname:<OPTIONAL PORT NUMBER>` in a web browser.

If a connection dies and you can't reconnect to your port because it's already allocated, run:

```bash
docker ps
docker stop <name of docker container with port forwarding>
```
