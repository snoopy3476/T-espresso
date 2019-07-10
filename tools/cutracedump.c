#include "../lib/cutrace_io.h"

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

#include <stdio.h>

#define die(...) do {\
  printf(__VA_ARGS__);\
  exit(1);\
} while(0)

const char* ACC_TYPE_NAMES[] = {
    "LD", "ST", "AT", "EXE", "RET"
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

  int64_t accesses = -1;
  while (trace_next(trace) == 0) {
    if (trace->new_kernel) {
      if (accesses > -1) {
        printf("  Total number of accesses: %" PRId64 "\n", accesses);
      }
      printf("Kernel name: %s\n", trace->kernel_name);
      accesses = 0;
    } else {
      trace_record_t *r = &trace->record;
      accesses += r->count;
      if (quiet) {
        continue;
      }
      if (r->count == 1) {
        if (r->type != 3 && r->type != 4) ///////////////////////////
	  printf("  type: %s, addr: 0x%" PRIx64 ", sm: %d, cta: (%2d,%2d,%2d), warp: %ld, size: %" PRIu32 "\n",
		 ACC_TYPE_NAMES[r->type], r->addr, r->smid,
	    r->ctaid.x, r->ctaid.y, r->ctaid.z, r->meta, r->size);
	else
	  printf("  type: %s, timer: %10" PRIu32 ", clock: %10" PRIu32 ", sm: %d, cta: (%2d,%2d,%2d), warp: %ld, size: %" PRIu32 "\n",
		 ACC_TYPE_NAMES[r->type], (uint32_t)(r->addr >> 32), (uint32_t)(r->addr & 2147483647), r->smid,
	    r->ctaid.x, r->ctaid.y, r->ctaid.z, r->meta, r->size);
      } else {
	  if (r->type != 3 && r->type != 4) /////////////////////////
	  printf("  type: %s, addr: 0x%" PRIx64 ", sm: %d, cta: (%2d,%2d,%2d), warp: %ld, size: %" PRIu32 ", count: %" PRIu16 "\n",
	    ACC_TYPE_NAMES[r->type], r->addr, r->smid,
	    r->ctaid.x, r->ctaid.y, r->ctaid.z, r->meta, r->size, r->count);
	else
	  printf("  type: %s, timer: %10" PRIu32 ", clock: %10" PRIu32 ", sm: %d, cta: (%2d,%2d,%2d), warp: %ld, size: %" PRIu32 ", count: %" PRIu16 "\n",
		 ACC_TYPE_NAMES[r->type], (uint32_t)(r->addr >> 32), (uint32_t)(r->addr & 2147483647), r->smid,
		 r->ctaid.x, r->ctaid.y, r->ctaid.z, r->meta, r->size, r->count);
      }
    }
  }
  if (trace_last_error != NULL) {
    printf("position: %zu\n", ftell(trace->file));
    die("%s\n", trace_last_error);
  }
  if (accesses > -1) {
    printf("  Total number of accesses: %" PRId64 "\n", accesses);
  }

  trace_close(trace);
}
