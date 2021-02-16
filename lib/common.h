#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif
  


#define RECORD_HEADER_UNIT (6)
#define RECORD_HEADER_UNIT_SIZE (sizeof(uint64_t))
#define RECORD_HEADER_SIZE \
  ((RECORD_HEADER_UNIT_SIZE) * (RECORD_HEADER_UNIT))
#define RECORD_DATA_UNIT_SIZE (sizeof(uint64_t))
#define RECORD_DATA_SIZE(data_len) \
  ((RECORD_DATA_UNIT_SIZE) * data_len)
#define RECORD_SIZE(data_len) \
  (RECORD_HEADER_SIZE + RECORD_DATA_SIZE(data_len))
#define RECORD_DATA_UNIT_MAX (32)
#define RECORD_SIZE_MAX (RECORD_SIZE(RECORD_DATA_UNIT_MAX))


// SLOT_SIZE: need to be power of two for performance
#define UNIT_SLOT_SIZE ((size_t) 0x80000) // total: 2MB
#define MULTI_BUF_COUNT (4)
#define SLOT_SIZE (UNIT_SLOT_SIZE * MULTI_BUF_COUNT)
//((size_t) 4096*(RECORD_SIZE_MAX) + (RECORD_SIZE_MAX))
// Number of slots must be power of two!
#define SLOTS_PER_DEV (16)

#define CACHELINE (128)

//#define CUPROF_RECBUF_MANAGED

//#define CUPROF_ODE_DISABLE
//#define CUPROF_CRW_DISABLE
//#define CUPROF_MULTI_BUF_DISABLE
//#define CUPROF_RMSYNC_DISABLE



// Reference type definition
  typedef struct {
    uint8_t* allocs_d;
    uint8_t* commits_d;
    uint8_t* flusheds_d;
    uint8_t* signals_d;
    uint8_t* records_d;
  } traceinfo_t;
  
  typedef struct {
    traceinfo_t info_d;
    uint8_t* flusheds_h;
    uint8_t* flusheds_old;
    uint8_t* signals_h;
    uint8_t* records_h;
  } traceinfo_host_t;
  
  typedef unsigned char byte;


#define LLGT_BIT_MASK(width)                    \
  (((uint64_t)1 << (width)) - 1)
#define LLGT_GET_BITFIELD(data, start, width)                   \
  ((((uint64_t)data) >> (start)) & (LLGT_BIT_MASK(width)))
#define LLGT_SET_BITFIELD(data, start, width)                   \
  ((((uint64_t)data) & (LLGT_BIT_MASK(width))) << (start))


  
// decoding records
  
#define RECORD_GET_NONZEROMASK(record)                  \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[0], 26, 38))
#define RECORD_GET_KERNID(record)                       \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[0], 16, 10))
#define RECORD_GET_INSTID(record)                       \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[0], 5, 11))
#define RECORD_GET_WARP_V(record)                       \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[0], 0, 5))
  
#define RECORD_GET_ACTIVEMASK(record)                   \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[1], 32, 32))
#define RECORD_GET_WRITEMASK(record)                    \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[1], 0, 32))

#define RECORD_GET_CTAX(record)                         \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[2], 32, 32))
#define RECORD_GET_CTAY(record)                         \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[2], 16, 16))
#define RECORD_GET_CTAZ(record)                         \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[2], 0, 16))
  
#define RECORD_GET_GRID(record)                 \
  (((uint64_t*)record)[3])

#define RECORD_GET_WARP_P(record)                       \
    (LLGT_GET_BITFIELD(((uint64_t*)record)[4], 32, 32))
#define RECORD_GET_SM(record)                           \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[4], 0, 32))
  
#define RECORD_GET_MSB(record)                          \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[5], 32, 32))
#define RECORD_GET_CLOCK(record)                        \
  (LLGT_GET_BITFIELD(((uint64_t*)record)[5], 0, 32))


#define RECORD_GET_DELTA(record, i)                                     \
  (LLGT_GET_BITFIELD(                                                   \
    ((uint64_t*)( ((uint64_t*)record) + RECORD_HEADER_UNIT)) [i],        \
    32, 32))
#define RECORD_GET_DATA(record, i)                                      \
  (LLGT_GET_BITFIELD(                                                   \
    ((uint64_t*)( ((uint64_t*)record) + RECORD_HEADER_UNIT)) [i],        \
    0, 32))

  
// encoding records

#define RECORD_SET_HEADER_0(nonzero_mask, kernid, instid, warpv)        \
  ((LLGT_SET_BITFIELD(nonzero_mask, 26, 38)) |                          \
   (LLGT_SET_BITFIELD(kernid, 16, 10)) |                                \
   (LLGT_SET_BITFIELD(instid, 5, 11)) |                                 \
   (LLGT_SET_BITFIELD(instid, 0, 5)))
#define RECORD_SET_HEADER_1(activemask, writemask)      \
  (LLGT_SET_BITFIELD(activemask, 32, 32) |              \
   (LLGT_SET_BITFIELD(writemask, 0, 32)))
#define RECORD_SET_HEADER_2(cta)                \
  ((uint64_t)(cta))
#define RECORD_SET_HEADER_3(grid)               \
  ((uint64_t)(grid))
#define RECORD_SET_HEADER_4(warpp, sm)          \
  (LLGT_SET_BITFIELD(warpp, 32, 32) |           \
   (LLGT_SET_BITFIELD(sm, 0, 32)))
#define RECORD_SET_HEADER_5(msb, clock)         \
  (LLGT_SET_BITFIELD(msb, 32, 32) |             \
   (LLGT_SET_BITFIELD(clock, 0, 32)))

#define RECORD_SET_DATA(delta, data)            \
  (LLGT_SET_BITFIELD(delta, 32, 32) |           \
   (LLGT_SET_BITFIELD(data, 0, 32)))

  

  enum RECORD_TYPE {
    RECORD_LOAD = 0,
    RECORD_STORE = 1,
    RECORD_ATOMIC = 2,
    RECORD_EXECUTE = 3,
    RECORD_RETURN = 4,
    RECORD_UNKNOWN = 5,
  };



  
#ifdef __cplusplus
}
#endif


  
#endif
