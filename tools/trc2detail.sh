#!/bin/sh

### prepend names on each field of cutracedump output ###

CUTRACEDUMP_BIN=cutracedump

prog='
$1=="K" \
{
        printf "trace_type=" $1 " kernel=" $2 "\n";
}

$1=="T" \
{
        printf "trace_type=" $1 " op=" $2 " grid=" $3 \
        " cta[x]=" $4 " cta[y]=" $5 " cta[z]=" $6 \
        " warpv=" $7 " cta_size=" $8 \
        " sm=" $9 " warpp=" $10 " clock=" $11 "\n";
}

$1=="M" \
{
        printf "trace_type=" $1 " op=" $2 " grid=" $3 \
        " cta[x]=" $4 " cta[y]=" $5 " cta[z]=" $6 \
        " warpv=" $7 " cta_size=" $8 \
        " sm=" $9 " warpp=" $10 " clock=" $11 \
        " req_size=" $12;

        for (i = 13; i < 45; i++)
        {
                printf " addr[" i-13 "]=" $i;
        }

        printf " inst_id=" $45 " kernel_name=" $46 " inst_line=" $47 " inst_col=" $48;


        printf " inst_src=\"";
        for (i = 49; i < NF; i++)
        {
                printf $i" ";
        }
        printf $i"\"\n"

}
'

"$CUTRACEDUMP_BIN" $@ | awk "$prog"
