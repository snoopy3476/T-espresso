#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif
  

// Size of a record in bytes, contents of a record:
// 32 bit meta info, 32bit size, 64 bit address, 64 bit cta id
#define RECORD_MAX_SIZE 304 //56
// 6M buffer, devided into 4 parallel slots.
// Buffers: SLOTS_PER_STREAM_IN_A_DEV * RECORDS_PER_SLOT * RECORD_MAX_SIZE
// Absolute minimum is the warp size, all threads in a warp must collectively
// wait or be able to write a record
#define SLOT_SIZE ((size_t)4096*RECORD_MAX_SIZE)
// Number of slots must be power of two!
#define SLOTS_PER_STREAM_IN_A_DEV (64)

#define CACHELINE (128)

#define CUPROF_RECBUF_MAPPED



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


#define DIV_ROUND_UP(dividend, divisor)                         \
  ((dividend)/(divisor) + ((dividend)%(divisor) ? 1 : 0))


#define RECORD_HEADER_UNIT 6
#define RECORD_HEADER_SIZE (RECORD_HEADER_UNIT * 8)
#define RECORD_SIZE(addr_len)          \
  (((RECORD_HEADER_SIZE) + (8 * (addr_len))))

#define RECORD_GET_ALEN(record)                 \
  (((uint64_t*)record)[0] >> 56)
#define RECORD_GET_INSTID(record)               \
  ((((uint64_t*)record)[0] >> 32) & 0xFFFFFF)
#define RECORD_GET_KERNID(record)               \
  ((((uint64_t*)record)[0] >> 8) & 0xFFFFFF)
#define RECORD_GET_WARP_V(record)               \
  (((uint64_t*)record)[0] & 0xFF)

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
  
  // #define RECORD_GET_REQ_SIZE(record) ((((uint64_t*)record)[4] >> 48) & 0xFFFF)
#define RECORD_GET_CLOCK(record)                \
  (((uint64_t*)record)[4])
  
#define RECORD_GET_ACTIVE(record)             \
    ((((uint64_t*)record)[5] >> 32) & 0xFFFFFFFF)
#define RECORD_GET_MSB(record)                \
  (((uint64_t*)record)[5] & 0xFFFFFFFF)
  
  
#define RECORD_ADDR_PTR(record, idx)            \
  (((uint8_t*)record) + RECORD_SIZE(idx))
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


#define RECORD_SET_INIT_IDX_0(addr_len, instid, kernid, warpv)          \
  ((((uint64_t)(addr_len)) << 56) |                                     \
   (((uint64_t)(instid) & 0xFFFFFF) << 32) |                            \
   (((uint64_t)(kernid) & 0xFFFFFF) << 8) |                             \
   ((uint64_t)(warpv) & 0xFF))
#define RECORD_SET_INIT_IDX_1(cta)              \
    ((uint64_t)(cta))
#define RECORD_SET_INIT_IDX_2(grid)             \
  ((uint64_t)(grid))
#define RECORD_SET_INIT_IDX_3(warpp, sm)        \
  ((((uint64_t)(warpp)) << 32) |                \
   ((uint64_t)(sm) & 0xFFFFFFFF))
#define RECORD_SET_INIT_IDX_4(clock)            \
  ((uint64_t)(clock))
#define RECORD_SET_INIT_IDX_5(msb, active)         \
  (((uint64_t)(msb) & 0xFFFFFFFF00000000) |        \
   (((uint64_t)(active)) & 0xFFFFFFFF))
  

//#define RECORD_SET_INIT_OPT(addr_len, type, instid, kernid, warpv,    \
//                            cta,                                        \
//                            grid,                                       \
//                            warpp, sm,                                  \
//                            req_size, clock)                            \
//  {                                                                     \
//    RECORD_SET_INIT_IDX_0(addr_len, type, instid, kernid, warpv),       \
//      RECORD_SET_INIT_IDX_1(cta),                                       \
//      RECORD_SET_INIT_IDX_2(grid),                                      \
//      RECORD_SET_INIT_IDX_3(warpp, sm),                                 \
//      RECORD_SET_INIT_IDX_4(req_size, clock)                            \
//      }


//#define RECORD_SET_INIT(addr_len, type, instid, kernid, warpv,      \
//                        cta_x, cta_y, cta_z,                    \
//                        grid,                                   \
//                        warpp, sm,                              \
//                        req_size, clock)                        \
//  RECORD_SET_INIT_OPT(addr_len, type, instid, kernid, warpv,    \
//                      (((uint64_t)(cta_x)) << 32) |             \
//                      (((uint64_t)(cta_y) & 0xFFFF) << 16) |    \
//                      ((uint64_t)(cta_z) & 0xFFFF),             \
//                      grid,                                     \
//                      warpp, sm,                                \
//                      req_size, clock)


    typedef struct {
      uint64_t data[5];
    } record_header_t;
  
  typedef struct {
    uint64_t data[DIV_ROUND_UP(RECORD_MAX_SIZE, 8)];
  } record_t;



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
