#ifndef __TRACE_IO_H__
#define __TRACE_IO_H__


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <unistd.h>
#include <fcntl.h>
#include "common.h"



#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#elif __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif



#ifdef __cplusplus
extern "C" {
#endif


#define NAME_UNKNOWN "(unknown)"
  
  static const char TRACE_HEADER_PREFIX[] = "__CUPROF_TRACE__";
  static const char TRACE_HEADER_POSTFIX[] = "__CUPROF_TRACE__END__";
  static const char* trace_last_error = NULL;


// under 16 recommended
#define TRACE_FILENAME_MAXLEN_BITS (12)
#define TRACE_FILENAME_MAXLEN_BYTES (((TRACE_FILENAME_MAXLEN_BITS)+8-1)/8) // ceil
#define TRACE_FILENAME_MAXLEN (1 << (TRACE_FILENAME_MAXLEN_BITS))

// under 16 recommended
#define TRACE_KERNELNAME_MAXLEN_BITS (12)
#define TRACE_KERNELNAME_MAXLEN_BYTES (((TRACE_KERNELNAME_MAXLEN_BITS)+8-1)/8) // ceil
#define TRACE_KERNELNAME_MAXLEN (1 << (TRACE_KERNELNAME_MAXLEN_BITS))

#define MIN(x, y) ((x) < (y) ? (x) : (y))
  
/****************
 * Trace Struct *
 ****************/
  
#define TRACEFILE_BUF_SIZE (SLOT_SIZE * 4)
#define TRACE_HEADER_BUF_SIZE_UNIT (1024 * 1024)

#define TRACE_HEADER_INST_META_SIZE 5
  
  
  typedef struct {
    int file;
    uint64_t buf_commits;
    unsigned char* buf;
  } tracefile_base_t;

  typedef tracefile_base_t* tracefile_t;

  
  
  typedef struct {
    uint32_t x;
    uint16_t y;
    uint16_t z;
  } cta_t;
  
  typedef struct {
    uint32_t id;
    uint32_t type;
    uint64_t meta[TRACE_HEADER_INST_META_SIZE];
    uint32_t row;
    uint32_t col;
    uint32_t filename_len;
    const char* filename;
  } trace_header_inst_t;
  
  static trace_header_inst_t empty_inst = {
    0,
    RECORD_UNKNOWN,
    {0},
    0,
    0,
    sizeof(NAME_UNKNOWN)+1
  };

  typedef struct {
    uint32_t insts_count;
    uint32_t kernel_name_len;
    char kernel_name[TRACE_KERNELNAME_MAXLEN];
    
    trace_header_inst_t insts[1]; // managed as flexible length member
  } trace_header_kernel_t;

  static trace_header_kernel_t empty_kernel = {
    0,
    sizeof(NAME_UNKNOWN)+1,
    NAME_UNKNOWN,
    {{0}}
  };
  

    
  typedef struct {
  
    const trace_header_kernel_t* kernel_info;
    const trace_header_inst_t* inst_info;
    uint32_t warpv;
    
    uint32_t activemask;
    uint32_t writemask;
    
    cta_t ctaid;
    
    uint64_t grid;
    
    uint32_t warpp;
    uint32_t sm;
    
    uint32_t msb;
    uint32_t clock;

    uint64_t thread_data[RECORD_DATA_UNIT_MAX];
  } trace_record_t;

  typedef struct {
    tracefile_t tracefile;
    uint64_t kernel_count;
    uint64_t kernel_i;
    trace_header_kernel_t** kernel_accdat;

    cta_t grid_dim;
    uint16_t cta_size;
    char new_kernel;
    trace_record_t record;
  } trace_t;

  
#define CONST_MAX(x, y) ((x) > (y) ? (x) : (y))
#include <errno.h> ///////////////////////////////////////////
  

/******************
 * Trace File I/O *
 ******************/
  
  typedef enum {TRACEFILE_READ, TRACEFILE_WRITE} tracefile_mode_t;
  
  static inline tracefile_t tracefile_open(const char* filename, tracefile_mode_t mode) {
    tracefile_t return_val = (tracefile_t) malloc(sizeof(tracefile_base_t));
    if (return_val == NULL)
      return NULL;


    
    if (filename == NULL) {
      return_val->file = STDIN_FILENO;
    }
    else {
      return_val->file = open(filename,
                              (mode == TRACEFILE_WRITE)
                              ? (O_WRONLY | O_CREAT | O_APPEND | O_TRUNC) // | O_DIRECT)
                              : (O_RDONLY),
                              S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    }
    if (return_val->file == -1) {
      free(return_val);
      return NULL;
    }

    

    return_val->buf_commits = 0;
    if (mode == TRACEFILE_WRITE) {
      return_val->buf = (byte*) malloc(TRACEFILE_BUF_SIZE); //aligned_alloc(512, TRACEFILE_BUF_SIZE);
      if (return_val->buf == NULL) {
        close(return_val->file);
        free(return_val);
        return NULL;
      }
    }

    return return_val;
  }
  
#include <sys/time.h>
static inline double rttclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

  
  static inline int tracefile_close(tracefile_t tracefile) {
    
    if (tracefile == NULL)
      return 0;


    // if unwritten data remains in the buffer, flush to file
    if (tracefile->buf_commits > 0) {
      write(tracefile->file,
            tracefile->buf,
            tracefile->buf_commits);
    }
    
    int close_result = close(tracefile->file);
    if (close_result == -1)
      return 0;

    
    if (tracefile->buf != NULL) {
      free(tracefile->buf);
    }

    free(tracefile);

    return 1;
  }

  static inline int tracefile_read(tracefile_t tracefile,
                                   void* dest, size_t size) {
    return (read(tracefile->file, dest, size) == (ssize_t) size);
  }
  
  static inline int tracefile_write(tracefile_t tracefile,
                                    const void* src, size_t size) {

    int return_val = 1;
    
    // if overflow is expected after write, then flush to file first
    if (tracefile->buf_commits + size >= TRACEFILE_BUF_SIZE) {
      ssize_t write_size = write(tracefile->file,
                                 tracefile->buf,
                                 tracefile->buf_commits);
      return_val = (write_size == (ssize_t)tracefile->buf_commits);
      tracefile->buf_commits = 0;
    }

    // copy to tracefile buffer
    memcpy(tracefile->buf + tracefile->buf_commits, src, size);
    tracefile->buf_commits += size;
    return return_val;
  }


/**********
 * reader *
 **********/

  static void uint64_serialize(byte* buf, size_t* offset, uint64_t input) {

    size_t offset_val = 0;

    if (offset != NULL) {
      offset_val = *offset;
      *offset += sizeof(uint64_t);
    }
    
    *(uint64_t*)(buf + offset_val) = input;
  }
  
  static uint64_t uint64_deserialize(byte* buf, size_t* offset) {

    size_t offset_val = 0;

    if (offset != NULL) {
      offset_val = *offset;
      *offset += sizeof(uint64_t);
    }
    
    return *(uint64_t*)(buf + offset_val);
  }
  

  
  static byte* header_serialize(size_t* out_size,
                                trace_header_kernel_t* kernel_header) {
    
    
    // build header byte data
    if (!kernel_header) {
      return NULL;
    }

    int buf_size = TRACE_HEADER_BUF_SIZE_UNIT;
    byte* buf = (byte*) malloc(TRACE_HEADER_BUF_SIZE_UNIT);
    if (!buf) {
      return NULL;
    }
    
    size_t offset = 0; // write with padding for the header size field

    // inst count
    uint64_serialize(buf, &offset, kernel_header->insts_count);

    // kernel name length
    uint64_serialize(buf, &offset, kernel_header->kernel_name_len);
    
    memcpy(buf + offset,
           kernel_header->kernel_name,
           kernel_header->kernel_name_len); // kernel name
    offset += kernel_header->kernel_name_len;

    
    // inst info
    for (uint32_t i = 1; i <= kernel_header->insts_count; i++) {

      // realloc if overflow (worst case)
      if (offset > buf_size - (sizeof(trace_header_inst_t)+TRACE_FILENAME_MAXLEN)) {
        buf_size += TRACE_HEADER_BUF_SIZE_UNIT;
        byte* buf_new = (byte*) realloc(buf, buf_size);
        if (!buf_new) {
          free(buf);
          return NULL;
        }
        buf = buf_new;
      }

      
      trace_header_inst_t* inst_header = &kernel_header->insts[i];

      // inst id in kernel
      uint64_serialize(buf, &offset, inst_header->id);
      
      // inst type
      uint64_serialize(buf, &offset, inst_header->type);

      for (int meta_i = 0; meta_i < TRACE_HEADER_INST_META_SIZE; meta_i++)
        uint64_serialize(buf, &offset, inst_header->meta[i]);
      
      // inst row in file
      uint64_serialize(buf, &offset, inst_header->row);
      
      // inst col in file
      uint64_serialize(buf, &offset, inst_header->col);
      
      // inst filename length
      uint64_serialize(buf, &offset, inst_header->filename_len);
      
      memcpy(buf + offset,
             inst_header->filename,
             inst_header->filename_len); // inst filename
      offset += inst_header->filename_len;

    }
    
    
    //int_serialize(buf, 4, (uint32_t)offset); // kernel header size to the front
    *out_size = offset;
    
    
    byte* buf_new = (byte*) realloc(buf, offset);
    if (!buf_new) {
      free(buf);
      return NULL;
    }
    buf = buf_new;

    return buf;
  }


  static size_t header_deserialize(trace_header_kernel_t* kernel_header,
                                   byte* buf) {
    
    if (!kernel_header || !buf) {
      return 0;
    }

    printf("\nKERNEL\n");//////////

    size_t offset = 0;

    // kernel info //
    
    // inst count
    kernel_header->insts_count = uint64_deserialize(buf, &offset);
    
    // kernel name length
    kernel_header->kernel_name_len = uint64_deserialize(buf, &offset);
    kernel_header->kernel_name_len = MIN(TRACE_KERNELNAME_MAXLEN,
                                         kernel_header->kernel_name_len);
    //////////////////////// remove std::min in InstrumentDevice.cpp /////////////
    
    memcpy(kernel_header->kernel_name,
           buf + offset,
           kernel_header->kernel_name_len); // kernel name
    offset += kernel_header->kernel_name_len;
    //printf("kernel_name: %s\n", kernel_header->kernel_name);
      
    
    // inst info
    for (uint32_t i = 1; i <= kernel_header->insts_count; i++) {
      trace_header_inst_t* inst_header = &kernel_header->insts[i];

      // inst id in kernel
      inst_header->id = uint64_deserialize(buf, &offset);
      
      // inst type in kernel
      inst_header->type = uint64_deserialize(buf, &offset);
      
      // metadata for inst
      for (int meta_i = 0; meta_i < TRACE_HEADER_INST_META_SIZE; meta_i++)
        inst_header->meta[i] = uint64_deserialize(buf, &offset);
      
      // inst row in file
      inst_header->row = uint64_deserialize(buf, &offset);
      
      // inst col in file
      inst_header->col = uint64_deserialize(buf, &offset);
      
      // inst filename length
      inst_header->filename_len = uint64_deserialize(buf, &offset);
      
      char* filename = 
        (char*) malloc(sizeof(inst_header->filename_len));
      inst_header->filename = filename;
      memcpy(filename,
             buf + offset,
             inst_header->filename_len); // inst filename
      offset += inst_header->filename_len;
      if (i != inst_header->id) {
        trace_last_error = "failed to deserialize kernel header";
        return 0;
      }

      printf("\ninstid: %u\n", i);//////////
    }

    return offset;
  }





  static void trace_deserialize(uint8_t* record_serialized, trace_t* trace,
                                uint32_t data_count) {
    
    trace_record_t* record = &trace->record;
    uint64_t nonzero_mask = RECORD_GET_NONZEROMASK(record_serialized);
    uint8_t is_msb_diff = 0;

    
    // recover header data before non-zero conversion //
    
    uint64_t* buf_header = (uint64_t*) (record_serialized);
    uint64_t nonzero_mask_header =
      LLGT_GET_BITFIELD(nonzero_mask, 0, RECORD_HEADER_UNIT);

    for (int i = 0; i < RECORD_HEADER_UNIT; i++)
      if ((nonzero_mask_header & ((uint64_t)1 << i)) == 0)
        buf_header[i] = 0; // recover header

    

    // deserialize header //
    
    uint64_t kernid = RECORD_GET_KERNID(record_serialized);
    record->kernel_info = (kernid < trace->kernel_count) ?
      trace->kernel_accdat[kernid] :
      &empty_kernel;
    uint64_t instid = RECORD_GET_INSTID(record_serialized);
    record->inst_info = (instid < record->kernel_info->insts_count) ?
      &record->kernel_info->insts[instid] :
      &empty_inst;
    record->warpv = RECORD_GET_WARP_V(record_serialized);
    
    record->activemask = RECORD_GET_ACTIVEMASK(record_serialized);
    record->writemask = RECORD_GET_WRITEMASK(record_serialized);
    if (record->writemask == 0) {
      record->writemask = record->activemask;
      is_msb_diff = 1; // writemask 0 means msb was different
    }
    
    record->ctaid.x = RECORD_GET_CTAX(record_serialized);
    record->ctaid.y = RECORD_GET_CTAY(record_serialized);
    record->ctaid.z = RECORD_GET_CTAZ(record_serialized);
    
    record->grid = RECORD_GET_GRID(record_serialized);
    
    record->warpp = RECORD_GET_WARP_P(record_serialized);
    record->sm = RECORD_GET_SM(record_serialized);
    
    record->msb = RECORD_GET_MSB(record_serialized);
    record->clock = RECORD_GET_CLOCK(record_serialized);


    
    // recover thread data before non-zero conversion //
    
    uint64_t* buf_data = (uint64_t*) (record_serialized + RECORD_HEADER_SIZE);
    uint64_t nonzero_mask_data = 
      LLGT_GET_BITFIELD(nonzero_mask, RECORD_HEADER_UNIT, RECORD_DATA_UNIT_MAX);
    
    int record_serialized_i = 0;
    for (int record_i = 0; record_i < RECORD_DATA_UNIT_MAX; record_i++)
      if (record->writemask & (~nonzero_mask_data) & (LLGT_SET_BITFIELD(1, record_i, 1)))
        buf_data[record_serialized_i++] = 0; // recover zero data
    

    
    // deserialize data //
    
    if (!is_msb_diff) {
      record_serialized_i = 0;
      uint64_t delta = 0;
      uint64_t data = 0;
      
      
      for (int record_i = 0; record_i < RECORD_DATA_UNIT_MAX; record_i++) {
          
        if (record->writemask & (LLGT_SET_BITFIELD(1, record_i, 1))) {
          delta = RECORD_GET_DELTA(record_serialized, record_serialized_i);
          data = RECORD_GET_DATA(record_serialized, record_serialized_i);
          record_serialized_i++;
        }


        // write record_i-th thread data
        if (record->activemask & (LLGT_SET_BITFIELD(1, record_i, 1))) {
          record->thread_data[record_i] = 
            (LLGT_SET_BITFIELD(record->msb, 32, 32) |   \
             (LLGT_SET_BITFIELD(data, 0, 32)));
        }
        else {
          record->thread_data[record_i] = 0;
        }

        
        data += delta;
      }
    }
    else {
      for (int i = 0; i < RECORD_DATA_UNIT_MAX; i++) {
        record->thread_data[i] =
          (RECORD_GET_DELTA(record, i) << 32) |
          (RECORD_GET_DATA(record, i));
      }
    }
  }


  
  static trace_t* trace_open(const char* filename) {
    //int debug_count = 0;
    //printf("%d\n", debug_count++);//////////////////
    tracefile_t input_file;
    input_file = tracefile_open(filename, TRACEFILE_READ);
    
    //printf("%d\n", debug_count++);//////////////////
    
    byte delim_buf[CONST_MAX(sizeof(TRACE_HEADER_PREFIX),
                             sizeof(TRACE_HEADER_POSTFIX) + 5)];

    // check trace header prefix    
    if (! tracefile_read(input_file, delim_buf, sizeof(TRACE_HEADER_PREFIX)) ||
        memcmp(delim_buf, TRACE_HEADER_PREFIX, sizeof(TRACE_HEADER_PREFIX))
        != 0) {
      trace_last_error = "failed to read trace header";
      return NULL;
    }

    //printf("%d\n", debug_count++);//////////////////
    
    // get trace header length
    uint64_t accdat_len;
    if (! tracefile_read(input_file, &accdat_len, sizeof(accdat_len))) {
      trace_last_error = "failed to read trace header length";
      return NULL;
    }

    //printf("%d\n", debug_count++);//////////////////
    
    // read access data part from header
    byte* accdat = (byte*) malloc(accdat_len);
    if (!accdat) {
      trace_last_error = "failed to allocate memory";
      return NULL;
    }

    //printf("%d\n", debug_count++);//////////////////
    
    if (! tracefile_read(input_file, accdat, accdat_len)) {
      trace_last_error = "failed to read trace header";
      return NULL;
    }

    //printf("%d\n", debug_count++);//////////////////
    
    // allocate trace_t
    trace_t* res = (trace_t*) malloc(sizeof(trace_t));
    res->kernel_accdat = (trace_header_kernel_t**) malloc(sizeof(trace_header_kernel_t*) * (0x3FF + 1)); //////////// 

    
    //printf("%d\n", debug_count++);//////////////////
    
    // build memory access data for each kernels
    uint64_t kernel_count = 0;

    res->kernel_accdat[0] = &empty_kernel;
    
    //printf("%d\n", debug_count++);//////////////////
    
    for (uint32_t offset = 0; offset < accdat_len; kernel_count++) {
      
      //printf("\t%d\n", debug_count++);//////////////////

      
      size_t kernel_header_size = sizeof(trace_header_kernel_t) +
        sizeof(trace_header_inst_t) * uint64_deserialize(accdat, NULL);
      
      //printf("kernel_header_size: %u\n", kernel_header_size);///////////////
      //printf("\t%d\n", debug_count++);//////////////////
      
      trace_header_kernel_t* kernel_cur =
        (trace_header_kernel_t*) malloc(kernel_header_size);
      
      //printf("\t%d\n", debug_count++);//////////////////
      
      if (!kernel_cur) {
        trace_last_error = "failed to allocate memory";
        return NULL;
      }
      
      res->kernel_accdat[kernel_count+1] = kernel_cur;

      //printf("\t%d\n", debug_count++);//////////////////
      
      size_t kernel_data_size = header_deserialize(kernel_cur, accdat + offset);
      
      //printf("\t%d\n", debug_count++);//////////////////
      
      if (kernel_data_size == 0) {
        trace_last_error = "failed to deserialize kernel header";
        return NULL;
      }
      offset += kernel_data_size;
    }


    
    // check trace header postfix
    if (! tracefile_read(input_file, delim_buf, sizeof(TRACE_HEADER_POSTFIX)) ||
        memcmp(delim_buf, TRACE_HEADER_POSTFIX, sizeof(TRACE_HEADER_POSTFIX))
        != 0) {
      trace_last_error = "failed to read trace header";
      return NULL;
    }


    res->tracefile = input_file;
    res->kernel_count = kernel_count;
    res->kernel_i = (uint64_t)-1;
    res->new_kernel = 0;

    free(accdat);
    
    return res;
  }


  static int trace_next(trace_t* t) {
    uint8_t buf[RECORD_SIZE_MAX]; // mem space for addr_len == threads per warp
    // end of file, this is not an error
    if (! tracefile_read(t->tracefile, buf, RECORD_HEADER_SIZE)) {
      trace_last_error = NULL;
      return 1;
    }

    /////////// need to move below inside trace_deserialize ////////////
    uint32_t writemask = RECORD_GET_WRITEMASK(buf);
    uint32_t activemask = RECORD_GET_ACTIVEMASK(buf);
    uint32_t data_count = writemask ?
      __builtin_popcount(writemask) :
      __builtin_popcount(activemask);

    if (! tracefile_read(t->tracefile, buf + RECORD_HEADER_SIZE,
                         RECORD_SIZE(data_count) - RECORD_HEADER_SIZE)) {
      trace_last_error = "unable to read record";
      return 1;
    }
    
    trace_deserialize(buf, t, data_count);
      
    trace_last_error = NULL;
    return 0;
  }

  static void trace_close(trace_t* t) {

    for (uint64_t i_kern = 1; i_kern <= t->kernel_count; i_kern++) {
      
      trace_header_kernel_t* kern_header = t->kernel_accdat[i_kern];
      for (uint64_t i_inst = 0; i_inst < kern_header->insts_count; i_inst++) {
        free((char*)kern_header->insts[i_inst].filename);
      }
      
      free(kern_header);
    }
    free(t->kernel_accdat);

    if (t->tracefile->file != STDIN_FILENO)
      tracefile_close(t->tracefile);
    
    free(t);
  }

  

  

/**********
 * writer *
 **********/

  static tracefile_t trace_write_open(const char* filename) {
    
    tracefile_t tracefile = tracefile_open(filename, TRACEFILE_WRITE);
    
    if (tracefile == NULL) {
      trace_last_error = "file create error";
    }

    return tracefile;
  }
  
  static int trace_write_header(tracefile_t tracefile, const void* accdat, uint64_t accdat_len) {
  
    if (! tracefile_write(tracefile, TRACE_HEADER_PREFIX, sizeof(TRACE_HEADER_PREFIX))) {
      trace_last_error = "header prefix write error";
      return 1;
    }

    if (! tracefile_write(tracefile, &accdat_len, sizeof(accdat_len))) {
      trace_last_error = "header length write error";
      return 1;
    }

    if (! tracefile_write(tracefile, accdat, accdat_len)) {
      trace_last_error = "header accdat write error";
      return 1;
    }

    if (! tracefile_write(tracefile, TRACE_HEADER_POSTFIX, sizeof(TRACE_HEADER_POSTFIX))) {
      trace_last_error = "header postfix write error";
      return 1;
    }
  
    trace_last_error = NULL;
    return 0;
  }

  static int trace_write_kernel(tracefile_t tracefile, const char* name,
                                uint64_t grid_dim, uint16_t cta_size) {
    
    uint8_t name_len = strlen(name) & 0xFF;
    uint64_t header = ((uint64_t)name_len << 48) | ((uint64_t)cta_size << 32);

    uint8_t write_size = sizeof(header) + sizeof(grid_dim) + name_len;
    uint8_t* write_buf = (uint8_t*) malloc(write_size);
    uint8_t* buf_ptr = write_buf;
    memcpy(buf_ptr, &header, sizeof(header));
    buf_ptr += sizeof(header);
    memcpy(buf_ptr, &grid_dim, sizeof(grid_dim));
    buf_ptr += sizeof(grid_dim);
    memcpy(buf_ptr, name, name_len);

    if (! tracefile_write(tracefile, write_buf, write_size)) {
      trace_last_error = "write error";
      return 1;
    }
    
    free(write_buf);
    trace_last_error = NULL;

    return 0;
  }
  
  static int trace_write_close(tracefile_t tracefile) {
    return tracefile_close(tracefile);
  }


  
#ifdef __cplusplus
}
#endif




#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif __clang__
#pragma clang diagnostic pop
#endif




#endif
