#!/bin/bash

elf=${@: -1}

set -x
smartelf2hex.sh $elf > test.hex
$*
