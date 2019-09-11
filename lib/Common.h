#pragma once

#include <stdint.h>

// Reference type definition
typedef struct {
  uint8_t *allocs;
  uint8_t *commits;
  uint8_t *records;
  uint32_t slot_size;
} traceinfo_t;



#define RECORD_RAW_SIZE(addr_len) \
  ((32 + (12 * (addr_len))) * sizeof(char))

#define RECORD_GET_ALEN(record) \
  (((const uint64_t*)record)[0] >> 56)
#define RECORD_GET_TYPE(record) \
  ((((const uint64_t*)record)[0] >> 48) & 0xFF)
#define RECORD_GET_SMID(record) \
  ((((const uint64_t*)record)[0] >> 32) & 0xFFFF)
#define RECORD_GET_WARP(record) \
  (((const uint64_t*)record)[0] & 0xFFFFFFFF)

#define RECORD_GET_CTAX(record) \
  (((const uint64_t*)record)[1] >> 32)
#define RECORD_GET_CTAY(record) \
  ((((const uint64_t*)record)[1] >> 16) & 0xFFFF)
#define RECORD_GET_CTAZ(record) \
  (((const uint64_t*)record)[1] & 0xFFFF)
  
#define RECORD_GET_CLOCK(record) \
  (((const uint64_t*)record)[2])
  
#define RECORD_GET_SIZE(record) \
  (((const uint64_t*)record)[3] >> 32)
#define RECORD_GET_INSTID(record) \
  (((const uint64_t*)record)[3] & 0xFFFFFFFF)

  
#define RECORD_ADDR_PTR(record, idx) \
  (((uint8_t*)record) + RECORD_RAW_SIZE(idx))
#define RECORD_META_PTR(record, idx) \
  (RECORD_ADDR_PTR(record, idx) + 8)


#define RECORD_ADDR(record, idx) \
  (*(uint64_t*)RECORD_ADDR_PTR(record, idx))
#define RECORD_ADDR_META(record, idx) \
  (*(uint32_t*)RECORD_META_PTR(record, idx))
#define RECORD_GET_COUNT(record, idx) \
  ((int8_t)((*(const uint32_t*)RECORD_META_PTR(record, idx)) & 0xFF))
#define RECORD_GET_OFFSET(record, idx) \
  ((*(const int32_t*)RECORD_META_PTR(record, idx)) >> 8)

#define RECORD_SET_INIT(addr_len, type, smid, warp, \
                        cta_x, cta_y, cta_z, \
                        clock, \
                        size, instid) \
  ((record_t){ {\
      (((uint64_t)addr_len) << 56) | (((uint64_t)type & 0xFF) << 48) | \
        (((uint64_t)smid & 0xFFFF) << 32) | ((uint64_t)warp & 0xFFFFFFFF), \
        (((uint64_t)cta_x) << 32) | (((uint64_t)cta_y & 0xFFFF) << 16) | ((uint64_t)cta_z & 0xFFFF), \
        ((uint64_t)clock), \
        (((uint64_t)size) << 32) | ((uint64_t)instid) & 0xFFFFFFFF,  \
        } \
  })

typedef struct record_t {
  uint64_t data[6];
} record_t;


  

#define RECORD_TYPE_SHIFT (28)
enum RECORD_TYPE {
  RECORD_LOAD = 0,
  RECORD_STORE = 1,
  RECORD_ATOMIC = 2,
  RECORD_EXECUTE = 3,
  RECORD_RETURN = 4,
  RECORD_UNKNOWN = 5,
};

// Size of a record in bytes, contents of a record:
// 32 bit meta info, 32bit size, 64 bit address, 64 bit cta id
//#define RECORD_SIZE (24)
#define RECORD_SIZE (48) //RECORD_RAW_SIZE(1)
// 6M buffer, devided into 4 parallel slots.
// Buffers: SLOTS_NUM * SLOTS_SIZE * RECORD_SIZE
// Absolute minimum is the warp size, all threads in a warp must collectively
// wait or be able to write a record
#define SLOTS_SIZE (256 * 1024)
// Number of slots must be power of two!
#define SLOTS_NUM (8)

#define CACHELINE (64)

#ifdef INCLUDE_LLVM_MEMTRACE_STUFF

// functions need to be static because we link it into both host and device
// instrumentation

#include "llvm/IR/DerivedTypes.h"
#include "llvm/ADT/Twine.h"



static llvm::StructType* getTraceInfoType(llvm::LLVMContext &ctx) {
  using llvm::Type;
  using llvm::StructType;

  Type *fields[] = {
    Type::getInt8PtrTy(ctx),
    Type::getInt8PtrTy(ctx),
    Type::getInt8PtrTy(ctx),
    Type::getInt32Ty(ctx),
  };

  return StructType::create(fields, "traceinfo_t");
}



enum CuprofSymbolType {
  SYMBOL_TRACE = 0,
  SYMBOL_DATA_VAR = 1,
  SYMBOL_DATA_FUNC = 2,
  SYMBOL_END = 3,
};


#define CUPROF_ACCDAT_VAR "___cuprof_accdat_var"
#define CUPROF_ACCDAT_VARLEN "___cuprof_accdat_varlen"
#define CUPROF_ACCDAT_CTOR "___cuprof_accdat_ctor"
#define CUPROF_ACCDAT_DTOR "___cuprof_accdat_dtor"
const char * const SYMBOL_TYPE_STR[] = {
  "___cuprof_traceinfo_",
  "___cuprof_accdat_var_",
  "___cuprof_accdat_func_"
};



static std::string getSymbolNameForKernel(const llvm::Twine &kernelName,
                                          CuprofSymbolType type = SYMBOL_TRACE) {
  
  if (type >= SYMBOL_END || type < (CuprofSymbolType)0)
    type = (CuprofSymbolType)0;

  return (SYMBOL_TYPE_STR[type] + kernelName).str();
}

static std::string getSymbolNameForKernel(const std::string &kernelName,
                                          CuprofSymbolType type = SYMBOL_TRACE) {
  return getSymbolNameForKernel(llvm::Twine(kernelName), type);
}

static std::string getSymbolNameForKernel(const llvm::StringRef &kernelName,
                                          CuprofSymbolType type = SYMBOL_TRACE) {
  return getSymbolNameForKernel(llvm::Twine(kernelName), type);
}

#endif
