#!/bin/bash

PREFIX="#pragma once

const unsigned char device_utils[] = {

"

DATA=$(hexdump "$1" -e '/1 "0x%02x, "' -v)


POSTFIX="

};"


echo "$PREFIX""$DATA""$POSTFIX" > $2
