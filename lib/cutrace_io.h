#ifndef __CUDA_TRACE_READER_H__
#define __CUDA_TRACE_READER_H__

#ifdef __cplusplus__
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "Common.h"

/** 
 *
 * 0x00 signals "new kernel"
 *   2 Byte: length of kernel name
 *   n Byte: kernel name
 *
 * 0xFF signals "uncompressed record" (Version 2, Version 3)
 *   4 Byte: SM Id
 *   4 Byte: <4 bit: type of access> <28 bit: size of access>
 *   8 Byte: Address of access
 *   4 Byte: CTA Id X
 *   2 Byte: CTA Id Y
 *   2 Byte: CTA Id Z
 */

  static const char cuprof_header_prefix[] = "#CUPROFTRACE#";
  static const char cuprof_header_postfix[] = "\0\0\0\0";

  
  
  typedef struct trace_header_inst_t {
    uint32_t inst_id;
    uint32_t row;
    uint32_t col;
    uint8_t inst_filename_len;
    char inst_filename[256];
  } trace_header_inst_t;

  typedef struct trace_header_kernel_t {
    uint8_t kernel_name_len;
    char kernel_name[256];
    uint32_t insts_count;
    trace_header_inst_t ** inst_by_id;
    trace_header_inst_t insts[];
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
  
    uint32_t warp;
    uint32_t size;
    uint32_t instid;
  
    uint8_t type;
    uint8_t addr_len;
    uint8_t smid;
  
    trace_record_addr_t addr_unit[];
  } trace_record_t;

  typedef struct trace_t {
    FILE *file;
    uint64_t kernel_count;
    uint64_t kernel_i;
    trace_header_kernel_t ** kernel_accdat;
    uint16_t block_size;
    char new_kernel;
    trace_record_t record;
  } trace_t;
  

  const char *trace_last_error = NULL;

/******************************************************************************
 * reader
 *****************************************************************************/

#define TRACE_RECORD_SIZE(addr_len) (offsetof(trace_record_t, addr_unit) + sizeof(trace_record_addr_t) * (addr_len))
#define CONST_MAX(x, y) ((x) > (y) ? (x) : (y))


  


  size_t serialize_int(char * buf, size_t byte, uint32_t input) {
    for (size_t i = 0; i < byte; i++) {
      buf[i] = ( (uint8_t)input >> (i * 8) ) & 0xFF;
    }
    
    return byte;
  }
  
  size_t unserialize_int(uint32_t * output, size_t byte, char * buf) {
    uint32_t value = 0;
    for (size_t i = 0; i < byte; i++) {
      value += ((uint32_t)buf[i] & 0xFF) << (i * 8);
    }
    *output = value;
    
    return byte;
  }
  

  size_t get_max_header_size_after_packed(trace_header_kernel_t * kernel_header) {
    return 4 + 1 + 4 + 256 + kernel_header->insts_count * (1 + 4 + 4 + 4 + 256);
  }

  
  size_t header_pack(char *buf, trace_header_kernel_t * kernel_header) {

    // build header byte data
    if (!buf || !kernel_header) {
      fprintf(stderr, "cuprof: Failed to pack kernel header!\n");
      abort();
    }
    
    size_t offset = 0;

    // kernel info
    offset += serialize_int(buf + offset, 4, (uint32_t)-1); // kernel placeholder
    buf[offset++] = kernel_header->kernel_name_len; // kernel name length
    offset += serialize_int(buf + offset, 4, kernel_header->insts_count); // inst count
    
    memcpy(buf + offset, kernel_header->kernel_name, kernel_header->kernel_name_len); // kernel name
    offset += kernel_header->kernel_name_len;

    
    // inst info
    for (uint32_t i = 0; i < kernel_header->insts_count; i++) {
      trace_header_inst_t *inst_header = &kernel_header->insts[i];
      
      buf[offset++] = inst_header->inst_filename_len; // inst filename length
      offset += serialize_int(buf + offset, 4, inst_header->inst_id); // inst id in kernel
      offset += serialize_int(buf + offset, 4, inst_header->row); // inst row in file
      offset += serialize_int(buf + offset, 4, inst_header->col); // inst col in file
      
      memcpy(buf + offset, inst_header->inst_filename, inst_header->inst_filename_len); // inst filename
      offset += inst_header->inst_filename_len;

    }

    return offset;
  }


  size_t get_header_size_after_unpacked(char * buf) {
    uint32_t count;
    unserialize_int(&count, 4, buf + 5); // inst count
    return sizeof(trace_header_kernel_t) + (sizeof(trace_header_inst_t) * count);
  }

  size_t header_unpack(trace_header_kernel_t * kernel_header, char * packed_data) {
    
    if (!kernel_header || !packed_data) {
      fprintf(stderr, "cuprof: Failed to pack kernel header!\n");
      abort();
    }


    uint32_t value = 0;
    size_t offset = 0;

    // kernel info
    offset += unserialize_int(&value, 4, packed_data + offset); // kernel placeholder
    kernel_header->kernel_name_len = packed_data[offset++]; // kernel name length
    offset += unserialize_int(&kernel_header->insts_count, 4, packed_data + offset); // inst count
    
    memcpy(kernel_header->kernel_name, packed_data + offset, kernel_header->kernel_name_len); // kernel name
    offset += kernel_header->kernel_name_len;

    
    // inst info
    for (uint32_t i = 0; i < kernel_header->insts_count; i++) {
      trace_header_inst_t *inst_header = &kernel_header->insts[i];
      
      inst_header->inst_filename_len = packed_data[offset++]; // inst filename length
      offset += unserialize_int(&inst_header->inst_id, 4, packed_data + offset); // inst id of kernelcallheader
      offset += unserialize_int(&inst_header->row, 4, packed_data + offset); // inst row in file
      offset += unserialize_int(&inst_header->col, 4, packed_data + offset); // inst col in file
      
      memcpy(inst_header->inst_filename, packed_data + offset, inst_header->inst_filename_len); // inst filename
      offset += inst_header->inst_filename_len;
    }

    return offset;
  }



  
  trace_t *trace_open(FILE *f) {
    char delim_buf[CONST_MAX(sizeof(cuprof_header_prefix), sizeof(cuprof_header_postfix) + 5)];

    if (fread(delim_buf, sizeof(cuprof_header_prefix), 1, f) < 1 &&
        memcmp(delim_buf, cuprof_header_prefix, sizeof(cuprof_header_prefix)) != 0) {
      trace_last_error = "failed to read trace header";
      return NULL;
    }
    
    trace_t *res = (trace_t*) malloc(offsetof(trace_t, record) + TRACE_RECORD_SIZE(32));

    res->kernel_accdat = (trace_header_kernel_t **) malloc(sizeof(trace_header_kernel_t*) * 256);

    
    // build memory access data for each kernels
    uint64_t kernel_count = 0;
    for (kernel_count = 0;
         fread(delim_buf, 4, 1, f) == 1 && memcmp(delim_buf, cuprof_header_postfix, 4) != 0;
         kernel_count++) {

      // increase kernel_accdat alloc size if not enough
      if ((kernel_count & 0xFF) == 0xFF) {
        trace_header_kernel_t ** kernel_accdat_ptr = (trace_header_kernel_t**)
          realloc(res->kernel_accdat, sizeof(trace_header_kernel_t*) * 256 * ((kernel_count / 256) + 2));

        if (!kernel_accdat_ptr) {
          trace_last_error = "failed to allocate memory";
          return NULL;
        }
        res->kernel_accdat = kernel_accdat_ptr;
      }
      
      if (fread(delim_buf + 4, 5, 1, f) != 1) {
        trace_last_error = "failed to read trace header";
        return NULL;
      }


      
      // initialize kernel
      int validity = 1;
      uint32_t kernel_header_size = get_header_size_after_unpacked(delim_buf);
      res->kernel_accdat[kernel_count] = (trace_header_kernel_t*) malloc(sizeof(char) * kernel_header_size);
      trace_header_kernel_t * kernel_cur = res->kernel_accdat[kernel_count];
      
      kernel_cur->kernel_name_len = delim_buf[4]; // kernel name length
      unserialize_int(&kernel_cur->insts_count, 4, delim_buf + 5); // inst count
      uint32_t inst_id_max = 256;
      kernel_cur->inst_by_id = (trace_header_inst_t**) malloc(sizeof(trace_header_inst_t*) * 256); // inst by id alloc
      validity &= (fread(kernel_cur->kernel_name, kernel_cur->kernel_name_len, 1, f) == 1 ? 1 : 0); // kernel name
      //memcpy(kernel_cur->kernel_name, 

      
      // initialize each insts
      for (uint32_t i = 0; i < kernel_cur->insts_count; i++) {
        trace_header_inst_t * inst_cur = &kernel_cur->insts[i];
        char inst_buf[4];

        // inst filename length
        validity &= (fread(&inst_cur->inst_filename_len, 1, 1, f) == 1 ? 1 : 0);

        // inst id in kernel
        validity &= (fread(inst_buf, 4, 1, f) == 1 ? 1 : 0);
        unserialize_int(&inst_cur->inst_id, 4, inst_buf);
        
        // inst row in file
        validity &= (fread(inst_buf, 4, 1, f) == 1 ? 1 : 0);
        unserialize_int(&inst_cur->row, 4, inst_buf);

        // inst col in file
        validity &= (fread(inst_buf, 4, 1, f) == 1 ? 1 : 0);
        unserialize_int(&inst_cur->col, 4, inst_buf);

        // inst filename
        validity &= (fread(inst_cur->inst_filename, inst_cur->inst_filename_len, 1, f) == 1 ? 1 : 0);


        // map inst to inst_by_id
        if (inst_cur->inst_id > inst_id_max) {
          inst_id_max = 256 * ((inst_cur->inst_id / 256) + 1);
          trace_header_inst_t** inst_by_id_tmp = (trace_header_inst_t**)
            realloc(kernel_cur->inst_by_id, sizeof(trace_header_inst_t*) * inst_id_max);

          if (!inst_by_id_tmp) {
            trace_last_error = "failed to allocate memory";
            return NULL;
          }

          kernel_cur->inst_by_id = inst_by_id_tmp;
        }

        kernel_cur->inst_by_id[inst_cur->inst_id] = inst_cur;
      }

      if (!validity) {
        trace_last_error = "failed to read from file";
        return NULL;
      }

    }

    res->file = f;
    res->kernel_count = kernel_count;
    res->kernel_i = (uint64_t)-1;
    res->new_kernel = 0;
    return res;
  }

  void __trace_unpack(const record_t *buf, trace_record_t *record) {
    record->addr_len = RECORD_GET_ALEN(buf);
    record->type = RECORD_GET_TYPE(buf);
    record->smid = RECORD_GET_SMID(buf);
    record->warp = RECORD_GET_WARP(buf);
  
    record->ctaid.x = RECORD_GET_CTAX(buf);
    record->ctaid.y = RECORD_GET_CTAY(buf);
    record->ctaid.z = RECORD_GET_CTAZ(buf);

    record->clock = RECORD_GET_CLOCK(buf);
  
    record->size = RECORD_GET_SIZE(buf);
    record->instid = RECORD_GET_INSTID(buf);

    for (uint8_t i = 0; i < record->addr_len; i++) {
      record->addr_unit[i].addr = RECORD_ADDR(buf, i);
      record->addr_unit[i].offset = RECORD_GET_OFFSET(buf, i);
      record->addr_unit[i].count = RECORD_GET_COUNT(buf, i);
    }
  }

  void __trace_pack(const trace_record_t *record, record_t *buf) {
  
    *buf = RECORD_SET_INIT(record->addr_len, record->type, record->smid, record->warp,
                           record->ctaid.x, record->ctaid.y, record->ctaid.z, record->clock, record->size, record->instid);
  
    for (uint8_t i = 0; i < record->addr_len; i++) {
      RECORD_ADDR(buf, i) = record->addr_unit[i].addr;
      RECORD_ADDR_META(buf, i) = (record->addr_unit[i].offset << 8) | ((int64_t)record->addr_unit[i].count & 0xFF);
    }
  }

// returns 0 on success
  int trace_next(trace_t *t) {
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
      uint16_t block_size = (buf[0] >> 32) & 0xFFFF;

      char kernel_name[256];
    
      if (fread(kernel_name, name_len, 1, t->file) != 1) {
        trace_last_error = "unable to read kernel name length";
        return 1;
      }

      for (uint64_t i = 0; i < t->kernel_count; i++) {
        if (t->kernel_accdat[i]->kernel_name_len == name_len &&
            memcmp(kernel_name, t->kernel_accdat[i]->kernel_name, name_len) == 0) {
          t->kernel_i = i;
          break;
        }
      }
      
      t->new_kernel = 1;
      t->block_size = block_size;
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
    
      __trace_unpack((record_t *)buf, &t->record);
      trace_last_error = NULL;
      return 0;
    }
  }

  int trace_eof(trace_t *t) {
    return feof(t->file);
  }

  void trace_close(trace_t *t) {

    for (uint64_t i = 0; i < t->kernel_count; i++) {
      free(t->kernel_accdat[i]->inst_by_id);
      free(t->kernel_accdat[i]);
    }
    free(t->kernel_accdat);
    
    fclose(t->file);
    free(t);
  }
  

  

/******************************************************************************
 * writer
 *****************************************************************************/

  int trace_write_header(FILE *f, const char * accdat, uint64_t accdat_len) {
  
    if (fwrite(cuprof_header_prefix, sizeof(cuprof_header_prefix), 1, f) < 1) {
      trace_last_error = "header write error";
      return 1;
    }

    if (fwrite(accdat, sizeof(char), accdat_len, f) < accdat_len) {
      trace_last_error = "header write error";
      return 1;
    }

    if (fwrite(cuprof_header_postfix, sizeof(char), 4, f) < 4) {
      trace_last_error = "header write error";
      return 1;
    }
  
    trace_last_error = NULL;
    return 0;
  }

  int trace_write_kernel(FILE *f, const char* name, uint16_t block_size) {
    uint8_t name_len = strnlen(name, 0xFF) & 0xFF;
    uint64_t header = ((uint64_t)name_len << 48) | ((uint64_t)block_size << 32);
  
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

  int trace_write_record(FILE *f, const trace_record_t *record) {
  
    char buf[TRACE_RECORD_SIZE(32)]; // mem space for addr_len == threads per warp
    __trace_pack(record, (record_t *)buf);
    if (fwrite(buf, RECORD_RAW_SIZE(record->addr_len), 1, f) < 1) {
      trace_last_error = "write error";
      return 1;
    }
  
    return 0;
  }

#ifdef __cplusplus__
}
#endif


#endif
