#include "../lib/common.h"
#define FULL_MASK 0xFFFFFFFF
#define I32_MAX 0xFFFFFFFF


extern "C" {

/**************************************************** 
 *  void ___cuprof_trace();
 *
 *  Write trace data and associated info
 *  to the externally allocated areas.
 */
  __device__ __noinline__ void ___cuprof_trace(uint8_t* records, uint8_t* allocs, uint8_t* commits,
                                  uint8_t to_be_traced,
                                  uint64_t addr, uint64_t ctaid_serial,
                                  uint32_t instid, uint32_t warpv,
                                  uint32_t sm, uint32_t warpp,
                                  uint16_t req_size, uint8_t type) {
    uint64_t clock;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(clock));
    
    if (!to_be_traced)
      return;

    uint32_t laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));
    uint64_t grid;
    asm volatile ("mov.u64 %0, %%gridid;" : "=l"(grid));
    uint32_t active = __ballot_sync(FULL_MASK, 1);
    uint32_t lowest   = __ffs(active)-1;
    //uint32_t rlaneid = __popc(active << (32 - laneid));
    //uint32_t n_active = __popc(active);

    volatile uint32_t* valloc = (uint32_t*) allocs;
    volatile uint32_t* vcommit = (uint32_t*) commits;
    uint32_t rec_offset = I32_MAX;
    
    
    // allocate space in slot
    if (laneid == lowest) {

      // try to allocate until valid
      while(*valloc >= SLOTS_SIZE ||
            (rec_offset = atomicInc((uint32_t*)allocs, I32_MAX)) >= SLOTS_SIZE) {

        // if slot is over-allocated (not valid),
        // cancel the allocation and wait for flush
        if (rec_offset != I32_MAX) {
          atomicDec((uint32_t*)allocs, I32_MAX);
          rec_offset = I32_MAX;
        }
      }

      // write header at lowest lane
      record_header_t* rec_header =
        (record_header_t*) &(records[rec_offset * RECORD_SIZE]);
      *rec_header =
        (record_header_t) RECORD_SET_INIT_OPT(0, type, instid, warpv,
                                              ctaid_serial,
                                              grid,
                                              warpp, sm,
                                              req_size, clock);
    }


    // write requested addr for each lane
    rec_offset = __shfl_sync(FULL_MASK, rec_offset, lowest);
    uint64_t* rec_addr = (uint64_t*) &(records[(rec_offset) * RECORD_SIZE +
                                               WARP_RECORD_RAW_SIZE(laneid)]);
    *rec_addr = addr;

    // guarantee all writes before to be written to the 'records'
    __threadfence_system();


    // commit space in slot
    if (laneid == lowest) atomicInc((uint32_t*)commits, I32_MAX);
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
