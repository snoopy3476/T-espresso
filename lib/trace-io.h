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

  
/****************
 * Trace Struct *
 ****************/
  
#define TRACEFILE_BUF_SIZE (RECORD_SIZE * RECORDS_PER_SLOT * sizeof(byte) * 4)
  
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
    uint32_t row;
    uint32_t col;
    uint8_t inst_filename_len;
    char inst_filename[256];
  } trace_header_inst_t;

  typedef struct {
    uint8_t kernel_name_len;
    char kernel_name[256];
    uint32_t insts_count;
    
    trace_header_inst_t insts[1]; // managed as flexible length member
  } trace_header_kernel_t;

  static trace_header_kernel_t empty_kernel = {
    sizeof(NAME_UNKNOWN)+1, NAME_UNKNOWN, 0, {{0}}
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
  
    uint32_t sm;
    uint32_t warpp;
    uint32_t warpv;
    uint32_t req_size;
    uint32_t instid;
  
    uint8_t type;
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
                              ? (O_WRONLY | O_CREAT | O_APPEND | O_TRUNC)
                              : (O_RDONLY),
                              S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    }
    if (return_val->file == -1) {
      free(return_val);
      return NULL;
    }

    

    return_val->buf_commits = 0;
    if (mode == TRACEFILE_WRITE) {
      return_val->buf = (byte*) malloc(TRACEFILE_BUF_SIZE);
      if (return_val->buf == NULL) {
        close(return_val->file);
        free(return_val);
        return NULL;
      }
    }

    return return_val;
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
      return_val = (write(tracefile->file,
                          tracefile->buf,
                          tracefile->buf_commits)
                    == (ssize_t)tracefile->buf_commits);
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
  

  static size_t get_max_header_bytes_after_serialize(trace_header_kernel_t* kernel_header) {
    return 4 + 1 + 4 + 256 + kernel_header->insts_count * (1 + 4 + 4 + 4 + 256);
  }

  
  static size_t header_serialize(byte* buf, trace_header_kernel_t* kernel_header) {

    // build header byte data
    if (!buf || !kernel_header) {
      return 0;
    }
    
    size_t offset = 0;

    // kernel info
    offset += int_serialize(buf + offset, 4, (uint32_t)-1); // kernel placeholder
    buf[offset++] = kernel_header->kernel_name_len; // kernel name length
    offset += int_serialize(buf + offset, 4, kernel_header->insts_count); // inst count
    
    memcpy(buf + offset,
           kernel_header->kernel_name,
           kernel_header->kernel_name_len); // kernel name
    offset += kernel_header->kernel_name_len;

    
    // inst info
    for (uint32_t i = 0; i < kernel_header->insts_count; i++) {
      trace_header_inst_t* inst_header = &kernel_header->insts[i];
      
      buf[offset++] = inst_header->inst_filename_len; // inst filename length
      offset += int_serialize(buf + offset, 4, inst_header->instid); // inst id in kernel
      offset += int_serialize(buf + offset, 4, inst_header->row); // inst row in file
      offset += int_serialize(buf + offset, 4, inst_header->col); // inst col in file
      
      memcpy(buf + offset,
             inst_header->inst_filename,
             inst_header->inst_filename_len); // inst filename
      offset += inst_header->inst_filename_len;

    }

    return offset;
  }


  static size_t get_kernel_header_bytes_after_deserialize(byte* buf) {
    uint32_t count;
    int_deserialize(&count, 4, buf + 5); // inst count
    return sizeof(trace_header_kernel_t) + (sizeof(trace_header_inst_t) * (count));
  }

  static size_t header_deserialize(trace_header_kernel_t* kernel_header, byte* serial_data) {
    
    if (!kernel_header || !serial_data) {
      return 0;
    }


    uint32_t value = 0;
    size_t offset = 0;

    // kernel info
    {
      // kernel placeholder
      offset += int_deserialize(&value, 4, serial_data + offset);
      // kernel name length
      kernel_header->kernel_name_len = serial_data[offset++];
      // inst count
      offset += int_deserialize(&kernel_header->insts_count, 4,
                                serial_data + offset); 
    
      memcpy(kernel_header->kernel_name,
             serial_data + offset,
             kernel_header->kernel_name_len); // kernel name
      offset += kernel_header->kernel_name_len;
    }
    
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
      
      memcpy(inst_header->inst_filename,
             serial_data + offset,
             inst_header->inst_filename_len); // inst filename
      offset += inst_header->inst_filename_len;

      if (i != inst_header->instid) {
        trace_last_error = "failed to deserialize kernel header";
        return 0;
      }
    }

    return offset;
  }





  static void trace_serialize(trace_record_t* record, record_t* buf) {
    
    record_t data = {
      RECORD_SET_INIT(
        record->addr_len, record->type,
        record->instid, record->warpv,
        record->ctaid.x, record->ctaid.y, record->ctaid.z,
        record->grid,
        record->warpp, record->sm,
        record->req_size, record->clock
        )
    };
    *buf = data;

    for (uint8_t i = 0; i < record->addr_len; i++) {
      RECORD_ADDR(buf, i) = record->addr_unit[i].addr;
      RECORD_ADDR_META(buf, i) =
        (record->addr_unit[i].offset << 8) |
        ((int64_t)record->addr_unit[i].count & 0xFF);
    }
  }

  static void trace_deserialize(record_t* buf, trace_record_t* record) {
    record->addr_len = RECORD_GET_ALEN(buf);
    record->type = RECORD_GET_TYPE(buf);
    record->instid = RECORD_GET_INSTID(buf);
    record->warpv = RECORD_GET_WARP_V(buf);
  
    record->ctaid.x = RECORD_GET_CTAX(buf);
    record->ctaid.y = RECORD_GET_CTAY(buf);
    record->ctaid.z = RECORD_GET_CTAZ(buf);
    
    record->grid = RECORD_GET_GRID(buf);
    
    record->warpp = RECORD_GET_WARP_P(buf);
    record->sm = RECORD_GET_SM(buf);
    
    record->req_size = RECORD_GET_REQ_SIZE(buf);
    record->clock = RECORD_GET_CLOCK(buf);
    

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
        record->addr_unit[i].addr = WARP_RECORD_ADDR(buf, i);
        record->addr_unit[i].offset = 0;
        record->addr_unit[i].count = 1;
      }
    }
  }


  
  static trace_t* trace_open(const char* filename) {

    tracefile_t input_file;
    input_file = tracefile_open(filename, TRACEFILE_READ);
    
    
    byte delim_buf[CONST_MAX(sizeof(TRACE_HEADER_PREFIX),
                             sizeof(TRACE_HEADER_POSTFIX) + 5)];

    // check trace header prefix    
    if (! tracefile_read(input_file, delim_buf, sizeof(TRACE_HEADER_PREFIX)) ||
        memcmp(delim_buf, TRACE_HEADER_PREFIX, sizeof(TRACE_HEADER_PREFIX))
        != 0) {
      trace_last_error = "failed to read trace header";
      return NULL;
    }

    // get trace header length
    uint32_t accdat_len;
    if (! tracefile_read(input_file, delim_buf, sizeof(uint32_t))) {
      trace_last_error = "failed to read trace header length";
      return NULL;
    }
    int_deserialize(&accdat_len, 4, delim_buf);


    
    // read access data part from header
    byte* accdat = (byte*) malloc(accdat_len);
    if (!accdat) {
      trace_last_error = "failed to allocate memory";
      return NULL;
    }

    if (! tracefile_read(input_file, accdat, accdat_len)) {
      trace_last_error = "failed to read trace header";
      return NULL;
    }

    // allocate trace_t
    trace_t* res = (trace_t*) malloc(offsetof(trace_t, record) + TRACE_RECORD_SIZE(32));
    res->kernel_accdat = (trace_header_kernel_t**) malloc(sizeof(trace_header_kernel_t*) * 256);

    
    // build memory access data for each kernels
    uint64_t kernel_count = 0;

    res->kernel_accdat[0] = &empty_kernel;
    for (uint32_t offset = 0; offset < accdat_len; kernel_count++) {
      size_t kernel_header_size = get_kernel_header_bytes_after_deserialize(accdat + offset);
      trace_header_kernel_t* kernel_cur = (trace_header_kernel_t*) malloc(kernel_header_size);
      if (!kernel_cur) {
        trace_last_error = "failed to allocate memory";
        return NULL;
      }
      
      res->kernel_accdat[kernel_count+1] = kernel_cur;

      size_t kernel_data_size = header_deserialize(kernel_cur, accdat + offset);
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

    // Entry is a kernel
    if (ch == 0x00) {

      // read header
      uint8_t name_len = (buf[0] >> 48) & 0xFF;
      uint16_t cta_size = (buf[0] >> 32) & 0xFFFF;
      
      // read grid dim
      uint64_t grid_dim_raw;
      if (! tracefile_read(t->tracefile, &grid_dim_raw, 8)) {
        trace_last_error = NULL;
        return 1;
      }
      cta_t grid_dim = {
        (uint32_t) (grid_dim_raw & 0xFFFFFFFF),
        (uint16_t) ((grid_dim_raw >> 32) & 0xFFFF),
        (uint16_t) ((grid_dim_raw >> 48) & 0xFFFF)
      };

      // read kernel name
      byte kernel_name[256];
    
      if (! tracefile_read(t->tracefile, kernel_name, name_len)) {
        trace_last_error = "unable to read kernel name length";
        return 1;
      }

      // find kernel index for the kernel name
      t->kernel_i = 0; // set as unknown first
      for (uint64_t i = 1; i <= t->kernel_count; i++) {
        if (t->kernel_accdat[i]->kernel_name_len == name_len &&
            memcmp(kernel_name, t->kernel_accdat[i]->kernel_name, name_len) == 0) {
          t->kernel_i = i;
          break;
        }
      }
      
      t->new_kernel = 1;
      t->grid_dim = grid_dim;
      t->cta_size = cta_size;
      trace_last_error = NULL;
      return 0;
    }

    // Entry is a record
    else {
      t->new_kernel = 0;
      if (! tracefile_read(t->tracefile, buf+1, RECORD_RAW_SIZE(ch) - 8)) {
        trace_last_error = "unable to read record";
        return 1;
      }
    
      trace_deserialize((record_t*)buf, &t->record);

      // if kernel is unknown, set instid to 0
      if (t->kernel_i == 0)
        t->record.instid = 0;
      
      trace_last_error = NULL;
      return 0;
    }
  }

  static void trace_close(trace_t* t) {

    for (uint64_t i = 1; i <= t->kernel_count; i++) {
      free(t->kernel_accdat[i]);
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

  static int trace_write_record(tracefile_t tracefile, trace_record_t* record) {
  
    byte buf[TRACE_RECORD_SIZE(32)]; // mem space for addr_len == threads per warp
    trace_serialize(record, (record_t*)buf);
    if (! tracefile_write(tracefile, buf, RECORD_RAW_SIZE(record->addr_len))) {
      trace_last_error = "write error";
      return 1;
    }
  
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
