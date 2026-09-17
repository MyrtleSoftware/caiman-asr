# Training on heterogeneous CPUs

Modern desktop computers may have heterogeneous CPUs i.e. performance and
efficiency cores. If you launch the data-loader with the default number of
cores this can slow down your train as the pipeline will be bottle-necked by
the much slower efficiency cores. This effect can be quite pronounced. For
example, on an 13th Gen Intel(R) Core(TM) i7-13700K training with 24 loader
worker threads trains at 350 UTT/s but training on 8 cores runs close to 500
UTT/s! To determine the correct number of cores run `lstopo --of console`. Then
set `--loader_workers_per_gpu=` as appropriate (you may need to apt install
`hwloc`).
