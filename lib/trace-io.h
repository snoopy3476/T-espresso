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
  static const char TRACE_HEADER_POSTFIX[] = "\0\0\0\0";
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
    uint32_t instid;
    uint32_t inst_type;
    uint32_t meta;
    uint32_t row;
    uint32_t col;
    uint32_t inst_filename_len;
    char* inst_filename;
  } trace_header_inst_t;

  typedef struct {
    uint32_t insts_count;
    uint32_t kernel_name_len;
    char kernel_name[TRACE_KERNELNAME_MAXLEN];
    
    trace_header_inst_t insts[1]; // managed as flexible length member
  } trace_header_kernel_t;

  static trace_header_kernel_t empty_kernel = {
    0, sizeof(NAME_UNKNOWN)+1, NAME_UNKNOWN, {{0}}
  };
  

  typedef struct {
    uint64_t addr;
    int32_t offset;
    int8_t count;
  } trace_record_addr_t;
    
  typedef struct {
    
    cta_t ctaid;
    uint64_t clock;
    uint64_t grid;
  
    uint32_t type;
    uint32_t sm;
    uint32_t warpp;
    uint32_t instid;
    uint32_t kernid;
    uint32_t msb;
    uint32_t active;
    uint32_t meta;
  
    uint8_t warpv;
    uint8_t addr_len;
  
    trace_record_addr_t addr_unit[1]; // managed as flexible length member
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

  
#define TRACE_RECORD_SIZE(addr_len)                                     \
  (sizeof(trace_record_t) + sizeof(trace_record_addr_t) * (addr_len - 1))
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

  static size_t int_serialize(byte* buf, size_t size_byte, uint32_t input) {
    
    for (size_t i = 0; i < size_byte; i++) {
      buf[i] = (uint8_t) (input >> (i * 8)) & 0xFF;
    }
    
    return size_byte;
  }
  
  static size_t int_deserialize(uint32_t* output, size_t size_byte, byte* buf) {
    
    uint32_t value = 0;
    for (size_t i = 0; i < size_byte; i++) {
      value += ((uint32_t)buf[i] & 0xFF) << (i * 8);
    }
    *output = value;
    
    return size_byte;
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
    
    size_t offset = 4; // write with padding for the header size field

    // kernel info
    //offset += int_serialize(buf + offset, 4, (uint32_t)-1); // kernel header size
    offset += int_serialize(buf + offset, 4, kernel_header->insts_count); // inst count
    offset += int_serialize(buf + offset, TRACE_KERNELNAME_MAXLEN_BYTES,
                            MIN(TRACE_KERNELNAME_MAXLEN,
                                kernel_header->kernel_name_len)); // kernel name len
    
    memcpy(buf + offset,
           kernel_header->kernel_name,
           kernel_header->kernel_name_len); // kernel name
    offset += kernel_header->kernel_name_len;

    
    // inst info
    for (uint32_t i = 0; i < kernel_header->insts_count; i++) {

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
      
      offset += int_serialize(buf + offset, TRACE_FILENAME_MAXLEN_BYTES,
                              inst_header->inst_filename_len); // inst filename length
      offset += int_serialize(buf + offset, 4, inst_header->instid); // inst id
      offset += int_serialize(buf + offset, 4, inst_header->row); // inst row
      offset += int_serialize(buf + offset, 4, inst_header->col); // inst col
      
      memcpy(buf + offset,
             inst_header->inst_filename,
             inst_header->inst_filename_len); // inst filename
      offset += inst_header->inst_filename_len;

    }
    
    
    int_serialize(buf, 4, (uint32_t)offset); // kernel header size to the front
    *out_size = offset;
    
    
    byte* buf_new = (byte*) realloc(buf, offset);
    if (!buf_new) {
      free(buf);
      return NULL;
    }
    buf = buf_new;

    return buf;
  }


  static size_t get_kernel_header_bytes_after_deserialize(byte* buf) {
    uint32_t count;
    int_deserialize(&count, 4, buf + 5); // inst count
    return sizeof(trace_header_kernel_t) + (sizeof(trace_header_inst_t) * (count));
  }

  static size_t header_deserialize(trace_header_kernel_t* kernel_header,
                                   byte* serial_data) {
    
    if (!kernel_header || !serial_data) {
      return 0;
    }


    size_t offset = 0;

    // kernel info
    
    // kernel placeholder
    //uint32_t value = 0;
    //offset += int_deserialize(&value, 4, serial_data + offset);
    
    // inst count
    offset += int_deserialize(&kernel_header->insts_count, 4,
                              serial_data + offset);
    // kernel name length
    offset += int_deserialize(&kernel_header->kernel_name_len,
                              TRACE_KERNELNAME_MAXLEN_BYTES,
                              serial_data + offset);
    kernel_header->kernel_name_len = MIN(TRACE_KERNELNAME_MAXLEN,
                                         kernel_header->kernel_name_len);
    //////////////////////// remove std::min in InstrumentDevice.cpp /////////////
    
    memcpy(kernel_header->kernel_name,
           serial_data + offset,
           kernel_header->kernel_name_len); // kernel name
    offset += kernel_header->kernel_name_len;
      
    
    // inst info
    for (uint32_t i = 1; i <= kernel_header->insts_count; i++) {
      trace_header_inst_t* inst_header = &kernel_header->insts[i];

      // inst filename length
      inst_header->inst_filename_len = serial_data[offset++];
      // inst id in kernel
      offset += int_deserialize(&inst_header->instid, 4, serial_data + offset);
      // inst row in file
      offset += int_deserialize(&inst_header->row, 4, serial_data + offset);
      // inst col in file
      offset += int_deserialize(&inst_header->col, 4, serial_data + offset);
      
      inst_header->inst_filename =
        (char*) malloc(sizeof(inst_header->inst_filename_len));
      memcpy(inst_header->inst_filename,
             serial_data + offset,
             inst_header->inst_filename_len); // inst filename
      offset += inst_header->inst_filename_len;
      printf("%u, %u\n", i, inst_header->instid);
      if (i+1 != inst_header->instid) {
        trace_last_error = "failed to deserialize kernel header";
        return 0;
      }
    }

    return offset;
  }





//  static void trace_serialize(trace_record_t* record, record_t* buf) {
//  }

  static void trace_deserialize(record_t* buf, trace_record_t* record) {
    record->addr_len = RECORD_GET_ALEN(buf);
    record->kernid = RECORD_GET_KERNID(buf);
    record->instid = RECORD_GET_INSTID(buf);
    record->warpv = RECORD_GET_WARP_V(buf);
  
    record->ctaid.x = RECORD_GET_CTAX(buf);
    record->ctaid.y = RECORD_GET_CTAY(buf);
    record->ctaid.z = RECORD_GET_CTAZ(buf);
    
    record->grid = RECORD_GET_GRID(buf);
    
    record->warpp = RECORD_GET_WARP_P(buf);
    record->sm = RECORD_GET_SM(buf);
    
    record->clock = RECORD_GET_CLOCK(buf);

    record->msb = RECORD_GET_MSB(buf);
    record->active = RECORD_GET_ACTIVE(buf);
    

    if (record->addr_len > 0) {
      for (uint8_t i = 0; i < record->addr_len; i++) {
        record->addr_unit[i].addr = RECORD_ADDR(buf, i);
        record->addr_unit[i].offset = RECORD_GET_OFFSET(buf, i);
        record->addr_unit[i].count = RECORD_GET_COUNT(buf, i);
      }
    }
    else {
      record->addr_len = 32;
      for (uint8_t i = 0; i < record->addr_len; i++) {
        record->addr_unit[i].addr = RECORD_ADDR(buf, i);
        record->addr_unit[i].offset = 0;
        record->addr_unit[i].count = 1;
      }
    }
  }


  
  static trace_t* trace_open(const char* filename) {
    int debug_count = 0;
    printf("%d\n", debug_count++);//////////////////
    tracefile_t input_file;
    input_file = tracefile_open(filename, TRACEFILE_READ);
    
    printf("%d\n", debug_count++);//////////////////
    
    byte delim_buf[CONST_MAX(sizeof(TRACE_HEADER_PREFIX),
                             sizeof(TRACE_HEADER_POSTFIX) + 5)];

    // check trace header prefix    
    if (! tracefile_read(input_file, delim_buf, sizeof(TRACE_HEADER_PREFIX)) ||
        memcmp(delim_buf, TRACE_HEADER_PREFIX, sizeof(TRACE_HEADER_PREFIX))
        != 0) {
      trace_last_error = "failed to read trace header";
      return NULL;
    }

    printf("%d\n", debug_count++);//////////////////
    
    // get trace header length
    uint32_t accdat_len;
    if (! tracefile_read(input_file, delim_buf, sizeof(uint32_t))) {
      trace_last_error = "failed to read trace header length";
      return NULL;
    }
    int_deserialize(&accdat_len, 4, delim_buf);


    printf("%d\n", debug_count++);//////////////////
    
    // read access data part from header
    byte* accdat = (byte*) malloc(accdat_len);
    if (!accdat) {
      trace_last_error = "failed to allocate memory";
      return NULL;
    }

    printf("%d\n", debug_count++);//////////////////
    
    if (! tracefile_read(input_file, accdat, accdat_len)) {
      trace_last_error = "failed to read trace header";
      return NULL;
    }

    printf("%d\n", debug_count++);//////////////////
    
    // allocate trace_t
    trace_t* res = (trace_t*) malloc(offsetof(trace_t, record) + TRACE_RECORD_SIZE(32));
    res->kernel_accdat = (trace_header_kernel_t**) malloc(sizeof(trace_header_kernel_t*) * 256);

    
    printf("%d\n", debug_count++);//////////////////
    
    // build memory access data for each kernels
    uint64_t kernel_count = 0;

    res->kernel_accdat[0] = &empty_kernel;
    
    printf("%d\n", debug_count++);//////////////////
    
    for (uint32_t offset = 0; offset < accdat_len; kernel_count++) {
      
      printf("\t%d\n", debug_count++);//////////////////
      
      size_t kernel_header_size = get_kernel_header_bytes_after_deserialize(accdat + offset);
      
      printf("\t%d\n", debug_count++);//////////////////
      
      trace_header_kernel_t* kernel_cur = (trace_header_kernel_t*) malloc(kernel_header_size);
      
      printf("\t%d\n", debug_count++);//////////////////
      
      if (!kernel_cur) {
        trace_last_error = "failed to allocate memory";
        return NULL;
      }
      
      res->kernel_accdat[kernel_count+1] = kernel_cur;

      printf("\t%d\n", debug_count++);//////////////////
      
      size_t kernel_data_size = header_deserialize(kernel_cur, accdat + offset);
      
      printf("\t%d\n", debug_count++);//////////////////
      
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
    uint64_t buf[TRACE_RECORD_SIZE(32)/8+1]; // mem space for addr_len == threads per warp
    // end of file, this is not an error
    if (! tracefile_read(t->tracefile, buf, 8)) {
      trace_last_error = NULL;
      return 1;
    }
    
    uint8_t ch = (buf[0] >> 56) & 0xFF;

    if (! tracefile_read(t->tracefile, buf+1, RECORD_SIZE(ch) - 8)) {
      trace_last_error = "unable to read record";
      return 1;
    }
    
    trace_deserialize((record_t*)buf, &t->record);
      
    trace_last_error = NULL;
    return 0;
  }

  static void trace_close(trace_t* t) {

    for (uint64_t i_kern = 1; i_kern <= t->kernel_count; i_kern++) {
      
      trace_header_kernel_t* kern_header = t->kernel_accdat[i_kern];
      for (uint64_t i_inst = 0; i_inst < kern_header->insts_count; i_inst++) {
        free(kern_header->insts[i_inst].inst_filename);
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

    byte accdat_len_serialized[4] = {0};
    int_serialize(accdat_len_serialized, 4, accdat_len);
    if (! tracefile_write(tracefile, accdat_len_serialized, sizeof(uint32_t))) {
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

//  static int trace_write_record(tracefile_t tracefile, trace_record_t* record) {
//  
//    byte buf[TRACE_RECORD_SIZE(32)]; // mem space for addr_len == threads per warp
//    trace_serialize(record, (record_t*)buf);
//    if (! tracefile_write(tracefile, buf, RECORD_SIZE(record->addr_len))) {
//      trace_last_error = "write error";
//      return 1;
//    }
//  
//    return 0;
//  }

  
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
