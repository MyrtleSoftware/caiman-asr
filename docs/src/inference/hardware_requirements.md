# Hardware requirements
The following is a guide to the hardware requirements for the CAIMAN-ASR server.

## Host requirements
The CAIMAN-ASR server requires a host machine with the following specifications:
* Four CPU cores per VectorPath card
* 4 GB of memory per VectorPath card
* 100 GB of storage
* Ubuntu 22.04 LTS (recommended)


## Bandwidth
The server requies 500 Mbits/s of bandwidth per 1000 streams. For example, 2000 streams would require 1 Gbit/s of bandwidth.
The bandwidth is dependent on the number of streams and not the number of cards (one card can handle 2000 streams with the `base` model or 800 streams with the `large` model).
This figure has some headroom built in; the measured value on a reliable connection is 700 Mbits/s for 2000 streams.
