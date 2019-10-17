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
        printf "warpid " $7 " " $45 " " $8 " ";
        for (i=13; i < 45; i++)
        {
                if ($i != "(blank)") {printf $i " ";}
        }
        printf "\n";
}
'

"$CUTRACEDUMP_BIN" $@ | awk "$prog"
