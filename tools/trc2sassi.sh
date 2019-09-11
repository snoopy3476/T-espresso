#!/bin/sh

### convert .trc file to the sassi-like format ###

CUTRACEDUMP_BIN=cutracedump

prog='
$1=="K" \
{
        printf "kernel " $2 "\n";
}
$1=="M" \
{
        printf "warpid " $8 " " $43 " " $4 " ";
        for (i=11; i < 43; i++)
        {
                if ($i != "(blank)") {printf $i " ";}
        }
        printf "\n";
}
'

"$CUTRACEDUMP_BIN" $@ | awk "$prog"
