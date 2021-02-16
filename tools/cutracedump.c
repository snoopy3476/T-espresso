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

#include "../lib/trace-io.h"

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define die(...) do {                           \
    fprintf(stderr, __VA_ARGS__);               \
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
  fprintf(stderr, "Usage: %s [trace_file]\n", program_name);
  fprintf(stderr, "\n");
  fprintf(stderr, "If a file is provided, reads a binary memory trace from it and\n");
  fprintf(stderr, "dumps it to stdout. If no file is provided, uses stdin.\n");
}

int main(int argc, char** argv) {

  char* input_filename = NULL;

  switch (argc) {
  case 2: {
    input_filename = argv[1];
  }
  case 1: {
    break;
  }
  default:
    usage("cutracedump");
    exit(1);
  }

  
  int quiet = (int) (getenv("QUIET") != NULL);

  trace_t* trace = trace_open(input_filename);

  if (trace == NULL) {
    die("%s", trace_last_error);
  }

  uint16_t cta_size = 0;
  cta_t grid_dim = {0};

  while (trace_next(trace) == 0) {

    trace_record_t* record = &trace->record;
    if (quiet) {
      continue;
    }
      
    const trace_header_kernel_t* kernel_info = record->kernel_info;
    const trace_header_inst_t* inst_info = record->inst_info;

    char trace_type;
    switch(inst_info->type) {
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
           " %" PRIu64
           " %" PRIu32 "/%" PRIu32
           " %" PRIu16 "/%" PRIu16
           " %" PRIu16 "/%" PRIu16
           " %" PRIu32
           " %" PRIu16
           " %" PRIu32 " %" PRIu32
           " %010" PRIu32,
           trace_type, OP_TYPE_NAMES[inst_info->type],
           record->grid,
           record->ctaid.x, grid_dim.x,
           record->ctaid.y, grid_dim.y,
           record->ctaid.z, grid_dim.z,
           record->warpv,
           cta_size,
           record->sm, record->warpp,
           record->clock);
      

    // print mem access info
    if (trace_type == 'M') {

      // access width
      printf(" %" PRIu64, inst_info->meta[0]);

      for (int i = 0; i < 32; i++) {

        uint64_t data_cur = record->thread_data[i];
        
        if (data_cur != 0) {
          printf(" %" PRIx64, data_cur);
        }
        else {
          printf(" (inactive)");
        }
      }

    }
    

    // inst id
    printf(" %" PRIu32 " %s",
           inst_info->id, kernel_info->kernel_name);

    // position at src file
    printf(" %" PRIu32 " %" PRIu32 " %s",
           inst_info->row, inst_info->col, inst_info->filename);
    

    printf("\n");

  }
  if (trace_last_error != NULL) {
    die("%s\n", trace_last_error);
  }

  trace_close(trace);
}
