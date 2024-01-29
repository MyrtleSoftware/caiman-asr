#!/usr/bin/env bash
# tar the contents of OUTPUT_DIR excluding any checkpoint files as these
# should not be shared with third-parties including Myrtle

: ${OUTPUT_DIR:="/results"}
: ${TAR_FILE:="logs_to_share.tar.gz"}

tar -czvf $TAR_FILE --exclude='*checkpoint.pt' $OUTPUT_DIR
