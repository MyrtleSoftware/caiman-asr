# Programming the Achronix Speedster7t FPGA

The bitstream that goes on the FPGA supports all the model architectures, and it only needs to be reprogrammed when Myrtle.ai releases an updated bitstream.
If you have received a demo system from Achronix or Myrtle.ai then the bitstream will likely already have been set up for you and you will not need to follow this step.

## Checking that the card has enumerated
You can check if the card has enumerated properly if lspci lists any devices with ID 12ba:0069

```
$ lspci -d 12ba:0069
25:00.0 Non-Essential Instrumentation [1300]: BittWare, Inc. Device 0069 (rev 01)
```

There should be a result for each card.
If the card has not enumerated properly, you may need to power cycle the machine.

## Flashing via JTAG
The board needs to have a JTAG cable connected to enable it to be flashed.
See the VectorPath [documentation](https://www.achronix.com/documentation/vectorpath-s7t-vg6-accelerator-card) for more information on how to connect the JTAG cable.

You also need to have the Achronix ACE software installed on the machine.
To acquire the [Achronix tool suite](https://www.achronix.com/product/fpga-design-tools-achronix), please contact Achronix support.
A license is not required, as "lab mode" is sufficient for flashing the FPGA.

Enter the ACE console:
```bash
sudo rlwrap /opt/ACE_9.1.1/Achronix-linux/ace -lab_mode -batch
```
Then run the following command:
```bash
jtag::get_connected_devices
```
This will list the devices connected via JTAG.
As above, there should be one device ID for each card.
If you have multiple devices connected you will need to repeat the programming step for all of them.

Set the `jtag_id` variable to the device ID (X) of the card you want to program:
```bash
set jtag_id X
```
Then run the following commands to program the card:
```bash
spi::program_bitstream config2 bitstream_page0.flash 1 -offset 0 -device_id $jtag_id -switch30
spi::program_bitstream config2 bitstream.flash 4 -offset 4096 -device_id $jtag_id -switch30
```
Now power-cycle the machine and the card should be programmed. A reboot is not sufficient.
