#ifndef __TRACE_IO_H__
#define __TRACE_IO_H__


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
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

  
  static const char TRACE_HEADER_PREFIX[] = "__CUPROF_TRACE__";
  static const char TRACE_HEADER_POSTFIX[] = "\0\0\0\0";


  
  
  typedef struct trace_header_inst_t {
    uint32_t instid;
    uint32_t row;
    uint32_t col;
    uint8_t inst_filename_len;
    char inst_filename[256];
  } trace_header_inst_t;

  typedef struct trace_header_kernel_t {
    uint8_t kernel_name_len;
    char kernel_name[256];
    uint32_t insts_count;
    
    trace_header_inst_t insts[1]; // managed as flexible length member
  } trace_header_kernel_t;

  

  typedef struct trace_record_addr_t {
    uint64_t addr;
    int32_t offset;
    int8_t count;
  } trace_record_addr_t;
    
  typedef struct trace_record_t {
    
    struct {
      uint32_t x;
      uint16_t y;
      uint16_t z;
    } ctaid;
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

  typedef struct trace_t {
    FILE* file;
    uint64_t kernel_count;
    uint64_t kernel_i;
    trace_header_kernel_t** kernel_accdat;
    uint32_t cta_size;
    char new_kernel;
    trace_record_t record;
  } trace_t;

  

#define NAME_UNKNOWN "(unknown)"
  static trace_header_kernel_t empty_kernel = {
    sizeof(NAME_UNKNOWN)+1, NAME_UNKNOWN, 0, {{0}}
  };
  
  static const char* trace_last_error = NULL;


  
/**********
 * reader *
 **********/
  
#define TRACE_RECORD_SIZE(addr_len)                                     \
  (sizeof(trace_record_t) + sizeof(trace_record_addr_t) * (addr_len - 1))
#define CONST_MAX(x, y) ((x) > (y) ? (x) : (y))


/**********
 * reader *
 **********/

  static size_t int_serialize(char* buf, size_t byte, uint32_t input) {
    
    for (size_t i = 0; i < byte; i++) {
      buf[i] = (uint8_t) (input >> (i * 8)) & 0xFF;
    }
    
    return byte;
  }
  
  static size_t int_deserialize(uint32_t* output, size_t byte, char* buf) {
    
    uint32_t value = 0;
    for (size_t i = 0; i < byte; i++) {
      value += ((uint32_t)buf[i] & 0xFF) << (i * 8);
    }
    *output = value;
    
    return byte;
  }
  

  static size_t get_max_header_bytes_after_serialize(trace_header_kernel_t* kernel_header) {
    return 4 + 1 + 4 + 256 + kernel_header->insts_count * (1 + 4 + 4 + 4 + 256);
  }

  
  static size_t header_serialize(char* buf, trace_header_kernel_t* kernel_header) {

    // build header byte data
    if (!buf || !kernel_header) {
      fprintf(stderr, "cuprof: Failed to serialize kernel header!\n");
      abort();
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


  static size_t get_kernel_header_bytes_after_deserialize(char* buf) {
    uint32_t count;
    int_deserialize(&count, 4, buf + 5); // inst count
    return sizeof(trace_header_kernel_t) + (sizeof(trace_header_inst_t) * (count));
  }

  static size_t header_deserialize(trace_header_kernel_t* kernel_header, char* serialized_data) {
    
    if (!kernel_header || !serialized_data) {
      fprintf(stderr, "cuprof: Failed to deserialize kernel header!\n");
      abort();
    }


    uint32_t value = 0;
    size_t offset = 0;

    // kernel info
    offset += int_deserialize(&value, 4, serialized_data + offset); // kernel placeholder
    kernel_header->kernel_name_len = serialized_data[offset++]; // kernel name length
    offset += int_deserialize(&kernel_header->insts_count, 4,
                              serialized_data + offset); // inst count
    
    memcpy(kernel_header->kernel_name,
           serialized_data + offset,
           kernel_header->kernel_name_len); // kernel name
    offset += kernel_header->kernel_name_len;

    
    // inst info
    for (uint32_t i = 1; i <= kernel_header->insts_count; i++) {
      trace_header_inst_t* inst_header = &kernel_header->insts[i];
      
      inst_header->inst_filename_len = serialized_data[offset++]; // inst filename length
      offset += int_deserialize(&inst_header->instid, 4,
                                serialized_data + offset); // inst id in kernel
      offset += int_deserialize(&inst_header->row, 4,
                                serialized_data + offset); // inst row in file
      offset += int_deserialize(&inst_header->col, 4,
                                serialized_data + offset); // inst col in file
      
      memcpy(inst_header->inst_filename,
             serialized_data + offset,
             inst_header->inst_filename_len); // inst filename
      offset += inst_header->inst_filename_len;

      if (i != inst_header->instid) {
        fprintf(stderr, "cuprof: Failed to deserialize kernel header!\n");
        abort();
      }
    }

    return offset;
  }





  static void trace_serialize(trace_record_t* record, record_t* buf) {
    
    record_t data = {
      RECORD_SET_INIT(record->addr_len, record->type,
                      record->instid, record->warpv,
                      record->ctaid.x, record->ctaid.y, record->ctaid.z,
                      record->grid,
                      record->req_size, record->clock,
                      record->warpp, record->sm)
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
    
    record->req_size = RECORD_GET_REQ_SIZE(buf);
    record->clock = RECORD_GET_CLOCK(buf);
    
    record->warpp = RECORD_GET_WARP_P(buf);
    record->sm = RECORD_GET_SM(buf);
  

    for (uint8_t i = 0; i < record->addr_len; i++) {
      record->addr_unit[i].addr = RECORD_ADDR(buf, i);
      record->addr_unit[i].offset = RECORD_GET_OFFSET(buf, i);
      record->addr_unit[i].count = RECORD_GET_COUNT(buf, i);
    }
  }


  
  static trace_t* trace_open(FILE* f) {
    char delim_buf[CONST_MAX(sizeof(TRACE_HEADER_PREFIX),
                             sizeof(TRACE_HEADER_POSTFIX) + 5)];

    // check trace header prefix    
    if (fread(delim_buf, sizeof(TRACE_HEADER_PREFIX), 1, f) < 1 &&
        memcmp(delim_buf, TRACE_HEADER_PREFIX, sizeof(TRACE_HEADER_PREFIX)) != 0) {
      trace_last_error = "failed to read trace header";
      return NULL;
    }

    // get trace header length
    uint32_t accdat_len;
    if (fread(delim_buf, sizeof(uint32_t), 1, f) < 1) {
      trace_last_error = "failed to read trace header length";
      return NULL;
    }
    int_deserialize(&accdat_len, 4, delim_buf);


    
    // read access data part from header
    char* accdat = (char*) malloc(sizeof(char) * accdat_len);
    if (!accdat) {
      trace_last_error = "failed to allocate memory";
      return NULL;
    }

    if (fread(accdat, sizeof(char), accdat_len, f) < accdat_len) {
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
      offset += header_deserialize(kernel_cur, accdat + offset);
    }


    
    // check trace header postfix
    if (fread(delim_buf, sizeof(TRACE_HEADER_POSTFIX), 1, f) < 1 &&
        memcmp(delim_buf, TRACE_HEADER_POSTFIX, sizeof(TRACE_HEADER_POSTFIX)) != 0) {
      trace_last_error = "failed to read trace header";
      return NULL;
    }


    res->file = f;
    res->kernel_count = kernel_count;
    res->kernel_i = (uint64_t)-1;
    res->new_kernel = 0;

    free(accdat);
    
    return res;
  }


  static int trace_next(trace_t* t) {
    uint64_t buf[TRACE_RECORD_SIZE(32)/8+1]; // mem space for addr_len == threads per warp
    // end of file, this is not an error
    if (fread(buf, 8 * sizeof(char), 1, t->file) != 1) {
      trace_last_error = NULL;
      return 1;
    }
    uint8_t ch = (buf[0] >> 56) & 0xFF;

    // Entry is a kernel
    if (ch == 0x00) {
      uint8_t name_len = (buf[0] >> 48) & 0xFF;
      uint16_t cta_size = (buf[0] >> 32) & 0xFFFF;

      char kernel_name[256];
    
      if (fread(kernel_name, name_len, 1, t->file) != 1) {
        trace_last_error = "unable to read kernel name length";
        return 1;
      }

      t->kernel_i = 0; // set as unknown first
      for (uint64_t i = 1; i <= t->kernel_count; i++) {
        if (t->kernel_accdat[i]->kernel_name_len == name_len &&
            memcmp(kernel_name, t->kernel_accdat[i]->kernel_name, name_len) == 0) {
          t->kernel_i = i;
          break;
        }
      }
      
      t->new_kernel = 1;
      t->cta_size = cta_size;
      trace_last_error = NULL;
      return 0;
    }

    // Entry is a record
    else {
      t->new_kernel = 0;
      if (fread(buf+1, RECORD_RAW_SIZE(ch) - 8, 1, t->file) != 1) {
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

  static int trace_eof(trace_t* t) {
    return feof(t->file);
  }

  static void trace_close(trace_t* t) {

    for (uint64_t i = 1; i <= t->kernel_count; i++) {
      free(t->kernel_accdat[i]);
    }
    free(t->kernel_accdat);
    
    fclose(t->file);
    free(t);
  }

  

  

/**********
 * writer *
 **********/

  static int trace_write_header(FILE* f, const char* accdat, uint64_t accdat_len) {
  
    if (fwrite(TRACE_HEADER_PREFIX, sizeof(TRACE_HEADER_PREFIX), 1, f) < 1) {
      trace_last_error = "header prefix write error";
      return 1;
    }

    char accdat_len_serialized[4] = {0};
    int_serialize(accdat_len_serialized, 4, accdat_len);
    if (fwrite(accdat_len_serialized, sizeof(uint32_t), 1, f) < 1) {
      trace_last_error = "header length write error";
      return 1;
    }

    if (fwrite(accdat, sizeof(char), accdat_len, f) < accdat_len) {
      trace_last_error = "header accdat write error";
      return 1;
    }

    if (fwrite(TRACE_HEADER_POSTFIX, sizeof(TRACE_HEADER_POSTFIX), 1, f) < 1) {
      trace_last_error = "header postfix write error";
      return 1;
    }
  
    trace_last_error = NULL;
    return 0;
  }

  static int trace_write_kernel(FILE* f, const char* name, uint16_t cta_size) {
    uint8_t name_len = strlen(name) & 0xFF;
    uint64_t header = ((uint64_t)name_len << 48) | ((uint64_t)cta_size << 32);
  
    if (fwrite(&header, 8, 1, f) < 1) {
      trace_last_error = "write error";
      return 1;
    }
  
    if (fwrite(name, name_len, 1, f) < 1) {
      trace_last_error = "write error";
      return 1;
    }
    trace_last_error = NULL;
    return 0;
  }

  static int trace_write_record(FILE* f, trace_record_t* record) {
  
    char buf[TRACE_RECORD_SIZE(32)]; // mem space for addr_len == threads per warp
    trace_serialize(record, (record_t*)buf);
    if (fwrite(buf, RECORD_RAW_SIZE(record->addr_len), 1, f) < 1) {
      trace_last_error = "write error";
      return 1;
    }
  
    return 0;
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
