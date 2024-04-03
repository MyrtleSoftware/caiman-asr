# Licensing
Licenses are required for each FPGA, and each license is tied to a particular FPGA's unique identifier.
Licenses may also have a maximum version number and release date that they support.
Additional or replacement licenses can be purchased by contacting Myrtle.ai or Achronix.

The CAIMAN-ASR server can run in "CPU mode", where the FPGA is not used and all inference is done on the CPU.
This does not require a license and is useful for testing; however the throughput of the CPU is much lower.
For details of how to run this, see the [CAIMAN-ASR server](./caiman-asr_server.md) documentation.

The directory containing the license file(s) is passed as an argument to the `start_server` script.
