#!/bin/bash

elf=${@: -1}
hex=${elf/\.elf/.hex}

set -x
smartelf2hex.sh $elf > $hex
$*
