#include "../lib/Common.h"
#define FULL_MASK 0xFFFFFFFF


extern "C" {
  
  __device__ void ___cuprof_trace(uint8_t* records, uint8_t* allocs, uint8_t* commits,
                                  uint64_t addr, uint64_t clock, uint64_t cta_arg,
                                  uint32_t instid, uint32_t warpid_v,
                                  uint16_t acc_size, uint8_t type) {

    uint32_t laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));
    uint64_t gridid;
    asm volatile ("mov.u64 %0, %%gridid;" : "=l"(gridid));
    uint32_t smid;
    asm volatile ("mov.u32 %0, %%smid;" : "=r"(smid));
    uint32_t warpid_p;
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warpid_p));
    uint32_t active = __ballot_sync(FULL_MASK, 1);
    uint32_t rlaneid = __popc(active << (32 - laneid));
    uint32_t n_active = __popc(active);
    uint32_t lowest   = __ffs(active)-1;

    volatile uint32_t* valloc = (uint32_t*) allocs;
    volatile uint32_t* vcommit = (uint32_t*) commits;
    unsigned int id = 0;

    /*
    uint32_t warpid_v = (threadIdx.x +
                         threadIdx.y * blockDim.x +
                         threadIdx.z * blockDim.x * blockDim.y) / 32; /////
    */
    
    
    // slot allocation
    
    if (laneid == lowest) {
      while(*valloc > (SLOTS_SIZE - 32) ||
            (id = atomicAdd((uint32_t*)allocs, n_active)) > (SLOTS_SIZE - 32)) {

        // if slot is over-allocated, cancel allocation and wait for flush
        if (id) {
          atomicSub((uint32_t*)allocs, n_active);
          id = 0;
        }
      }
    }


    
    // record write
    
    uint32_t record_offset = __shfl_sync(FULL_MASK, id, lowest) + rlaneid;
    record_t* record = (record_t*) &(records[(record_offset) * RECORD_SIZE]);
    
    *record = (record_t) RECORD_SET_INIT_OPT(1, type, instid & 0xFFFF, warpid_v,
                                         cta_arg,
                                         gridid,
                                         warpid_p, smid,
                                         acc_size, clock);
    RECORD_ADDR(record, 0) = addr;
    RECORD_ADDR_META(record, 0) = 1;
    
    __threadfence_system();


    
    // slot commit
    
    if (laneid == lowest) atomicAdd((uint32_t*)commits, n_active);
  }

}
