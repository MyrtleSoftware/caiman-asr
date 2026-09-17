# TensorBoard

The training scripts write TensorBoard logs to /results during training.

To monitor training using TensorBoard, launch the port-forwarding TensorBoard container in another terminal:

```bash
./scripts/docker/launch_tb.sh <results_dir1> [results_dir2 ...] [--port PORT] [--samples NUM] [--reload_interval SECONDS]
```

If `--port` isn't passed then it defaults to port 6010. `--samples` is the
number of steps that TensorBoard will sample from the log and plot. It defaults
to 1000. `--reload_interval` sets how often the backend scans for new data; it defaults to 30 seconds to reduce I/O overhead.

Then navigate to `http://traininghostname:<OPTIONAL PORT NUMBER>` in a web browser.

If a connection dies and you can't reconnect to your port because it's already allocated, run:

```bash
docker ps
docker stop <name of docker container with port forwarding>
```
