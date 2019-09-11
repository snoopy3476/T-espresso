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
        printf "trace_type=" $1 " op=" $2 " sm=" $3 \
        " cta_size=" $4 " cta[x]=" $5 " cta[y]=" $6 " cta[z]=" $7 \
        " warp=" $8 " clock=" $9 "\n";
}

$1=="M" \
{
        printf "trace_type=" $1 " op=" $2 " sm=" $3 \
        " cta_size=" $4 " cta[x]=" $5 " cta[y]=" $6 " cta[z]=" $7 \
        " warp=" $8 " clock=" $9 \
        " req_size=" $10;

        for (i = 11; i < 43; i++)
        {
                printf " addr[" i-10 "]=" $i;
        }

        printf " inst_id=" $43 " kernel_name=" $44 " inst_line=" $45 " inst_col=" $46;


        printf " inst_src=\"";
        for (i = 47; i < NF; i++)
        {
                printf $i" ";
        }
        printf $i"\"\n"

}
'

"$CUTRACEDUMP_BIN" $@ | awk "$prog"
