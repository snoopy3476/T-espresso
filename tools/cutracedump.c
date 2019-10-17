/***
 **
 **  Output format description
 **
 **  [Kernel call]
 **  K <kernel_name>
 **
 **  [Thread scheduling]
 **  T <operation> <sm_id> <cta_size> <cta_id_x> <cta_id_y> <cta_id_z> <warp_id> <clock>
 **
 **  [Memory access]
 **  M <operation> <sm_id> <cta_size> <cta_id_x> <cta_id_y> <cta_id_z> <warp_id> <clock>
 **    <request_size> <request_address(1)> <request_address(2)> ... <request_address(32)>
 **    <instruction_id> <kernel_name_for_instruction_id>
 **    <instruction_line_in_src> <instruction_col_in_src>
 **    <filename_of_src> <filename_of_src (If filename contains the space char ' ')> ...
 **
 **/

#define WARP_SIZE 32

#include "../lib/TraceIO.h"

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

#include <stdio.h>

#define die(...) do {                           \
    printf(__VA_ARGS__);                        \
    exit(1);                                    \
  } while(0)

// <operation>
const char* OP_TYPE_NAMES[] = {
  
  "LD", // Load (Memory)
  "ST", // Store (Memory)
  "AT", // Atomic (Memory)
  
  "EX", // Execution (Thread)
  "RT", // Return (Thread)
  
  "??"  // Undefined
};

void usage(const char* program_name) {
  printf("Usage: %s [trace_file]\n", program_name);
  printf("\n");
  printf("If a file is provided, reads a binary memory trace from it and\n");
  printf("dumps it to stdout. If no file is provided, uses stdin.\n");
}

int main(int argc, char** argv) {
  FILE* input;

  switch (argc) {
  case 2: {
    input = fopen(argv[1], "r");
    if (input == NULL) {
      die("Unable to open file '%s', exiting\n", argv[1]);
    }
    break;
  }
  case 1: {
    input = stdin;
    break;
  }
  default:
    usage("cutracedump");
    exit(1);
  }

  
  int quiet = getenv("QUIET") != NULL;

  trace_t* trace = trace_open(input);

  if (trace == NULL) {
    die("%s", trace_last_error);
  }

  uint64_t cta_size = 0;

  trace_header_kernel_t* kernel_info;  
  while (trace_next(trace) == 0) {

    if (trace->new_kernel) {
      kernel_info = trace->kernel_accdat[trace->kernel_i];
      printf("K %s\n", kernel_info->kernel_name);
      cta_size = trace->cta_size;
    } else {
      trace_record_t* r = &trace->record;
      if (quiet) {
        continue;
      }

      char trace_type;
      switch(r->type) {
      case RECORD_EXECUTE:
      case RECORD_RETURN:
        trace_type = 'T';
        break;

      case RECORD_LOAD:
      case RECORD_STORE:
      case RECORD_ATOMIC:
        trace_type = 'M';
        break;

      default:
        trace_type = '?';
        break;
      }
      
        
      
      printf("%c %s"
             " %" PRIu64 " %" PRIu32 " %" PRIu16 " %" PRIu16 " %" PRIu32
             " %" PRIu64
             " %" PRIu32 " %" PRIu32
             " %015" PRIu64,
             trace_type, OP_TYPE_NAMES[r->type],
             r->grid, r->ctaid.x, r->ctaid.y, r->ctaid.z, r->warpv,
             cta_size,
             r->sm, r->warpp,
             r->clock);
      

      // print mem access info
      if (trace_type == 'M') {
        trace_header_inst_t* inst_info = &kernel_info->insts[r->instid];

        // size
        printf(" %" PRIu32, r->req_size);

        int8_t count_total = 0;
        for (int32_t addr_i = 0; addr_i < r->addr_len; addr_i++) {
          trace_record_addr_t* acc_addr = &r->addr_unit[addr_i];
          
          for (int32_t acc_i = 0; acc_i < acc_addr->count; acc_i++) {
            printf(" %" PRIx64, acc_addr->addr + (acc_addr->offset)*acc_i);
          }

          count_total += acc_addr->count;
        }
        
        // if less then WARP_SIZE access, fill with blank
        for (uint32_t count_remain = count_total;
             count_remain < WARP_SIZE; count_remain++) {
          printf(" #");
        }

        // inst id
        printf(" %" PRIu32 " %s",
               inst_info->instid, kernel_info->kernel_name);

        // position at src file
        if (inst_info->inst_filename_len > 0) {
          printf(" %" PRIu32 " %" PRIu32 " %s",
                 inst_info->row, inst_info->col, inst_info->inst_filename);
        }

      }

      printf("\n");

    }
  }
  if (trace_last_error != NULL) {
    printf("position: %zu\n", ftell(trace->file));
    die("%s\n", trace_last_error);
  }

  trace_close(trace);
}
