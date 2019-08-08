#include "../lib/cutrace_io.h"

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

#include <stdio.h>

#define die(...) do {                           \
    printf(__VA_ARGS__);                        \
    exit(1);                                    \
  } while(0)

const char* ACC_TYPE_NAMES[] = {
  "LD", "ST", "AT", "CALL", "RET", "??"
};

void usage(const char* program_name) {
  printf("Usage: %s [file]\n", program_name);
  printf("\n");
  printf("If a file is provided, reads a binary memory trace from it and\n");
  printf("dumps it to stdout. If no file is provided, uses stdin.\n");
}

int main(int argc, char** argv) {
  FILE *input;
  if (argc == 1) {
    input = stdin;
  } else if (argc == 2) {
    input = fopen(argv[1], "r");
    if (input == NULL) {
      die("Unable to open file '%s', exiting\n", argv[1]);
    }
  } else {
    usage("cutracedump");
    exit(1);
  }

  int quiet = getenv("QUIET") != NULL;

  trace_t *trace = trace_open(input);

  if (trace == NULL) {
    die("%s", trace_last_error);
  }

  //int64_t accesses = -1;
  uint16_t block_size = 0;
  
  while (trace_next(trace) == 0) {
    
    if (trace->new_kernel) {
      printf("\n\nkernel %s\n\n", trace->kernel_name);
      block_size = trace->block_size;
    } else {
      trace_record_t *r = &trace->record;
      if (quiet) {
        continue;
      }
      
      printf("warpid %" PRIu32 " %" PRIu32 " %" PRIu16, r->warp, r->instid, block_size);

      // print mem
      if (r->type == ACCESS_CALL || r->type == ACCESS_RETURN) {

        for (uint8_t i = 0; i < r->addr_len; i++) {
          trace_record_addr_t *acc_addr = &r->addr_unit[i];
          int64_t increment = acc_addr->offset;
          int8_t count = acc_addr->count;
        
          for (int8_t j = 0; j < count; j++) {
            printf(" %" PRIu64, (acc_addr->addr) + increment*j);
          }
        }

        printf(" \t|sm|%" PRIu8 "|\t|cta|%" PRIu32 "/%" PRIu16 "/%" PRIu16
               "|\t|type|%s|\t|clk|%020" PRIu64 "|\n",
               r->smid, r->ctaid.x, r->ctaid.y, r->ctaid.z,
               ACC_TYPE_NAMES[r->type],
               r->clock);
        
      } else {
        for (uint8_t i = 0; i < r->addr_len; i++) {
          trace_record_addr_t *acc_addr = &r->addr_unit[i];
          int64_t increment = acc_addr->offset;
          int8_t count = acc_addr->count;
        
          for (int8_t j = 0; j < count; j++) {
            printf(" %" PRIx64, (acc_addr->addr) + increment*j);
          }
        }
        
        printf(" \t|sm|%" PRIu8 "|\t|cta|%" PRIu32 "/%" PRIu16 "/%" PRIu16
               "|\t|type|%s|\t|clk|%020" PRIu64 "|\t|size|%" PRIu32 "|\t|kernel|%s|\n",
               r->smid, r->ctaid.x, r->ctaid.y, r->ctaid.z,
               ACC_TYPE_NAMES[r->type],
               r->clock, r->size, trace->kernel_name);
      }

    }
  }
  if (trace_last_error != NULL) {
    printf("position: %zu\n", ftell(trace->file));
    die("%s\n", trace_last_error);
  }

  trace_close(trace);
}
