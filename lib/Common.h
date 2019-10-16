#pragma once

#include <stdint.h>



// Size of a record in bytes, contents of a record:
// 32 bit meta info, 32bit size, 64 bit address, 64 bit cta id
#define RECORD_SIZE 56
// 6M buffer, devided into 4 parallel slots.
// Buffers: SLOTS_NUM * SLOTS_SIZE * RECORD_SIZE
// Absolute minimum is the warp size, all threads in a warp must collectively
// wait or be able to write a record
#define SLOTS_SIZE (256 * 1024)
// Number of slots must be power of two!
#define SLOTS_NUM 32

#define CACHELINE 64





// Reference type definition
typedef struct {
  uint8_t *allocs;
  uint8_t *commits;
  uint8_t *records;
  uint32_t slot_size;
} traceinfo_t;


#define DIV_ROUND_UP(dividend, divisor)                         \
  ((dividend)/(divisor) + ((dividend)%(divisor) ? 1 : 0))




#define RECORD_RAW_SIZE(addr_len)               \
  ((40 + (12 * (addr_len))) * sizeof(char))

#define RECORD_GET_ALEN(record)                 \
  (((uint64_t*)record)[0] >> 56)
#define RECORD_GET_TYPE(record)                 \
  ((((uint64_t*)record)[0] >> 48) & 0xFF)
#define RECORD_GET_INSTID(record)               \
  ((((uint64_t*)record)[0] >> 32) & 0xFFFF)
#define RECORD_GET_WARP_V(record)               \
  (((uint64_t*)record)[0] & 0xFFFFFFFF)

#define RECORD_GET_CTAX(record)                 \
  (((uint64_t*)record)[1] >> 32)
#define RECORD_GET_CTAY(record)                 \
  ((((uint64_t*)record)[1] >> 16) & 0xFFFF)
#define RECORD_GET_CTAZ(record)                 \
  (((uint64_t*)record)[1] & 0xFFFF)
  
#define RECORD_GET_GRID(record)                 \
  (((uint64_t*)record)[2])

#define RECORD_GET_WARP_P(record)               \
  (((uint64_t*)record)[3] >> 32)
#define RECORD_GET_SM(record)                   \
  (((uint64_t*)record)[3] & 0xFFFFFFFF)
  
#define RECORD_GET_REQ_SIZE(record)             \
  ((((uint64_t*)record)[4] >> 48) & 0xFFFF)
#define RECORD_GET_CLOCK(record)                \
  (((uint64_t*)record)[4] & 0xFFFFFFFFFFFF)
  
  
#define RECORD_ADDR_PTR(record, idx)            \
  (((uint8_t*)record) + RECORD_RAW_SIZE(idx))
#define RECORD_META_PTR(record, idx)            \
  (RECORD_ADDR_PTR(record, idx) + 8)


#define RECORD_ADDR(record, idx)                \
  (*(uint64_t*)RECORD_ADDR_PTR(record, idx))
#define RECORD_ADDR_META(record, idx)           \
  (*(uint32_t*)RECORD_META_PTR(record, idx))
#define RECORD_GET_COUNT(record, idx)                           \
  ((int8_t)((*(uint32_t*)RECORD_META_PTR(record, idx)) & 0xFF))
#define RECORD_GET_OFFSET(record, idx)                  \
  ((*(int32_t*)RECORD_META_PTR(record, idx)) >> 8)

#define RECORD_SET_INIT_OPT(addr_len, type, instid, warp_v,             \
                            cta,                                        \
                            grid,                                       \
                            warp_p, sm,                                 \
                            clock,                                      \
                            req_size)                                   \
  {                                                                     \
  (((uint64_t)addr_len) << 56) | (((uint64_t)type & 0xFF) << 48) |      \
  (((uint64_t)instid & 0xFFFF) << 32) | ((uint64_t)warp_v & 0xFFFFFFFF), \
    ((uint64_t)cta),                                                    \
    ((uint64_t)grid),                                                   \
    (((uint64_t)warp_p) << 32) | ((uint64_t)sm & 0xFFFFFFFF),           \
    (((uint64_t)req_size & 0xFFFF) << 48) | (((uint64_t)clock) & 0xFFFFFFFFFFFF), \
    }


#define RECORD_SET_INIT(addr_len, type, instid, warp_v,         \
                        cta_x, cta_y, cta_z,                    \
                        grid,                                   \
                        warp_p, sm,                             \
                        req_size,                               \
                        clock)                                  \
  RECORD_SET_INIT_OPT(addr_len, type, instid, warp_v,           \
                      (((uint64_t)cta_x) << 32) |               \
                      (((uint64_t)cta_y & 0xFFFF) << 16) |      \
                      ((uint64_t)cta_z & 0xFFFF),               \
                      grid,                                     \
                      warp_p, sm,                               \
                      req_size,                                 \
                      clock)


  typedef struct record_t {
  uint64_t data[DIV_ROUND_UP(RECORD_SIZE, 8)];
} record_t;



enum RECORD_TYPE {
  RECORD_LOAD = 0,
  RECORD_STORE = 1,
  RECORD_ATOMIC = 2,
  RECORD_EXECUTE = 3,
  RECORD_RETURN = 4,
  RECORD_UNKNOWN = 5,
};


#ifdef INCLUDE_LLVM_CUPROF_TRACE_STUFF

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

#endif
