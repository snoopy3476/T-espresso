#include "../lib/common.h"
//#include <stdio.h>

extern "C" {
  
  __device__ uint64_t ___cuprof_grid_allocs_count[0x10000][2]; // 0x10000 == 65536

/**************************************************** 
 *  void ___cuprof_trace();
 *
 *  Write trace data and associated info
 *  to the externally allocated areas.
 */
  __device__ __noinline__ void ___cuprof_trace(uint32_t* alloc, uint32_t* commit,
                                               uint32_t* count, uint8_t* records,
                                               uint64_t addr,
                                               uint64_t grid, uint64_t ctaid_serial,
                                               uint32_t warpv, uint32_t lane,
                                               uint32_t instid,
                                               uint32_t sm, uint32_t warpp,
                                               uint16_t req_size,
                                               uint8_t type, uint8_t to_be_traced) {
    
    if (!to_be_traced)
      return;

    uint64_t clock;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(clock));
    uint32_t active;
    asm volatile ("activemask.b32 %0;" : "=r"(active));
    uint32_t lowest = __ffs(active)-1;

    volatile uint32_t* valloc = alloc;
    volatile uint32_t* vcommit = commit;
    volatile uint32_t* vcount = count;
    
    uint32_t rec_offset = UINT32_MAX;
    
    
    // allocate space in slot
    if (lane == lowest) {
      
      // try to allocate until available

      /*
      while (*valloc >= RECORDS_PER_SLOT ||
             (rec_offset = atomicInc(alloc, UINT32_MAX)) >= RECORDS_PER_SLOT) {
        if (rec_offset != UINT32_MAX) {
          atomicDec(alloc, UINT32_MAX);
          rec_offset = UINT32_MAX;
        }
      }
      */
        
      while (true) {

        // if allocation is full, then wait for flush
        while (*valloc >= RECORDS_PER_SLOT) {
          
          // if flush req sent, and all flushed on host
          if ((*valloc == RECORDS_PER_SLOT) && (*vcommit == UINT32_MAX) &&
              (*vcount == 0)) {
            
            if (atomicInc(commit, UINT32_MAX) == UINT32_MAX) {
              __threadfence(); // guarantee resetting vcommit after valloc
              *valloc = 0;
            }
            else {
              atomicDec(commit, UINT32_MAX);
            }
            
          }
        }

        // if overallocated, then cancel the allocation and wait for flush
        if ((rec_offset = atomicInc(alloc, UINT32_MAX)) >= RECORDS_PER_SLOT) {
          atomicDec(alloc, UINT32_MAX);
        }
        else {
          break;
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
    rec_offset = __shfl_sync(active, rec_offset, lowest);
    uint64_t* rec_addr = (uint64_t*) &(records[(rec_offset) * RECORD_SIZE +
                                               WARP_RECORD_RAW_SIZE(lane)]);
    *rec_addr = addr;


    // guarantee all writes before to be written to the 'records'
    __threadfence_system();
    /*
    if (lane == lowest) {
      if (atomicInc(commit, UINT32_MAX) == RECORDS_PER_SLOT - 1) {
        *vcount = *vcommit;
        __threadfence_system();
        while(*vcount);
        *vcommit = 0;
        __threadfence();
        *valloc = 0;
      }
    }
    */
    
    // commit space in slot, and send full signal to the host
    if (lane == lowest) {
      if (atomicInc(commit, UINT32_MAX) == RECORDS_PER_SLOT - 1) {
        *vcount = RECORDS_PER_SLOT; // request flush to host
        //printf("DEV_vcommit: %u\n", *vcommit);
        __threadfence_system();
        *vcommit = UINT32_MAX; // request sent successfully
      }
    }
    
  }

  

/**************************************************** 
 *  void ___cuprof_trace_ret();
 *
 *  Flush vcommit to count (host)
 */
  __device__ void ___cuprof_trace_ret(uint32_t* commit, uint32_t* count,
                                      uint32_t lane) {
    
    uint32_t active;
    asm volatile ("activemask.b32 %0;" : "=r"(active));
    uint32_t lowest = __ffs(active)-1;

    volatile uint32_t* vcommit = commit;
    volatile uint32_t* vcount = count;

    if (lane == lowest) {
      __threadfence();
      uint32_t rec_count = *vcommit;

      // if request not sent at the point of return, then send request
      if (rec_count != UINT32_MAX) {
        atomicMax(count, rec_count);
      }
      // guarantee write before return
      __threadfence_system();
    }

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
