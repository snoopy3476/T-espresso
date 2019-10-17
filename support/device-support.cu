#include "../lib/Common.h"
#define FULL_MASK 0xFFFFFFFF


extern "C" {

/**************************************************** 
 *  void ___cuprof_trace();
 *
 *  Write trace data and associated info
 *  to the externally allocated areas.
 */
  __device__ void ___cuprof_trace(uint8_t* records, uint8_t* allocs, uint8_t* commits,
                                  uint8_t to_be_traced,
                                  uint64_t addr, uint64_t ctaid_serial,
                                  uint32_t instid, uint32_t warpv,
                                  uint32_t sm, uint32_t warpp,
                                  uint16_t acc_size, uint8_t type) {
    uint64_t clock;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(clock));
    
    if (!to_be_traced)
      return;

    uint32_t laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));
    uint64_t grid;
    asm volatile ("mov.u64 %0, %%gridid;" : "=l"(grid));
    uint32_t active = __ballot_sync(FULL_MASK, 1);
    uint32_t rlaneid = __popc(active << (32 - laneid));
    uint32_t n_active = __popc(active);
    uint32_t lowest   = __ffs(active)-1;

    volatile uint32_t* valloc = (uint32_t*) allocs;
    volatile uint32_t* vcommit = (uint32_t*) commits;
    unsigned int id = 0;
    
    
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

    *record = (record_t) RECORD_SET_INIT_OPT(1, type, instid, warpv, ctaid_serial,
                                             grid, warpp, sm, acc_size, clock);
    RECORD_ADDR(record, 0) = addr;
    RECORD_ADDR_META(record, 0) = 1;
    
    __threadfence_system();


    // slot commit
    if (laneid == lowest) atomicAdd((uint32_t*)commits, n_active);
  }

  

/**************************************************** 
 *  void ___cuprof_filter();
 *
 *  Check if current thread is to be traced, 
 *  with given thread-constant vars (grid, cta, warpv).
 *
 *  Called only once in a thread, when the thread starts.
 */
  __device__ void ___cuprof_filter(uint8_t* to_be_traced, uint64_t* filter_grid,
                                   uint64_t* filter_cta, uint32_t* filter_warpv,
                                   uint8_t filter_grid_count,
                                   uint8_t filter_cta_count,
                                   uint8_t filter_warpv_count,
                                   uint64_t ctaid_serial, uint32_t warpv) {
    
    uint64_t grid;
    asm volatile ("mov.u64 %0, %%gridid;" : "=l"(grid));

    // count == 0 (do not filter): default value is true (!0)
    // otherwise: default value is false (!count), and set to true if exists in filter
    uint8_t to_be_traced_per_type[3] = {
      !filter_grid_count,
      !filter_cta_count,
      !filter_warpv_count
    };
    

    // check grid filter
    for (uint32_t i = 0; i < filter_grid_count; i++)
      if (filter_grid[i] == grid)
        to_be_traced_per_type[0] = 1;

    // check cta filter
    for (uint32_t i = 0; i < filter_cta_count; i++)
      if (filter_cta[i] == ctaid_serial)
        to_be_traced_per_type[1] = 1;

    // check warpv filter
    for (uint32_t i = 0; i < filter_warpv_count; i++)
      if (filter_warpv[i] == warpv)
        to_be_traced_per_type[2] = 1;

    // combine per_type with AND conditions
    uint8_t result = 1;
    for (uint32_t i = 0; i < 3; i++)
      if (!to_be_traced_per_type[i])
        result = 0;

    *to_be_traced = result;
  }


  
/**************************************************** 
 *  void ___cuprof_filter_volatile();
 *
 *  Check if current thread is to be traced, 
 *  with given volatile vars (sm, warpp).
 *
 *  Called on every trace, iff the filter of sm, warpp is set.
 */
  __device__ void ___cuprof_filter_volatile(uint8_t* to_be_traced,
                                            uint32_t* filter_sm, uint32_t* filter_warpp,
                                            uint8_t filter_sm_count,
                                            uint8_t filter_warpp_count,
                                            uint32_t sm, uint32_t warpp) {

    // count == 0 (do not filter): default value is true (!0)
    // otherwise: default value is false (!count), and set to true if exists in filter
    uint8_t to_be_traced_per_type[2] = {
      !filter_sm_count,
      !filter_warpp_count
    };
    
    // check sm filter
    for (uint32_t i = 0; i < filter_sm_count; i++)
      if (filter_sm[i] == sm)
        to_be_traced_per_type[0] = 1;

    // check warpp filter
    for (uint32_t i = 0; i < filter_warpp_count; i++)
      if (filter_warpp[i] == warpp)
        to_be_traced_per_type[1] = 1;

    // combine per_type with AND conditions
    uint8_t result = 1;
    for (uint32_t i = 0; i < 2; i++)
      if (!to_be_traced_per_type[i])
        result = 0;

    *to_be_traced = result;
  }
}
