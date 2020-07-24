#include "../lib/common.h"

extern "C" {

  
/**************************************************** 
 *  void ___cuprof_trace();
 *
 *  Write trace data and associated info
 *  to the externally allocated areas.
 */
  __device__ __noinline__ void ___cuprof_trace(uint32_t* alloc, uint32_t* commit,
                                               uint32_t* flushed, uint32_t* signal,
                                               uint8_t* records, uint64_t addr,
                                               uint64_t grid, uint64_t ctaid_serial,
                                               uint32_t warpv, uint32_t lane,
                                               uint32_t instid, uint32_t kernid,
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
    uint32_t rlane_id = __popc(active << (32 - lane));

    volatile uint32_t* alloc_v = alloc;
    volatile uint32_t* commit_v = commit;
    volatile uint32_t* flushed_v = flushed;
    volatile uint32_t* signal_v = signal;
    
    uint32_t rec_offset;
    
    
    // allocate space in slot
    if (lane == lowest) {


      //uint32_t counter = 0;
      do {
        while (*alloc_v >= RECORDS_PER_SLOT) {
          //counter++;
          //if ((counter & 0xFFFFF) == 0xFFFFF)
          //  printf("%u (%u)\n", *alloc_v, counter++);
        }

        //printf("%u, %u, %u\n", *alloc_v, *commit_v, *signal_v);//////////////
      } while ((rec_offset = atomicInc(alloc, UINT32_MAX)) >= RECORDS_PER_SLOT);

      // write header at lowest lane
      /*
      record_header_t* rec_header =
        (record_header_t*) &(records[rec_offset * RECORD_SIZE]);
      *rec_header =
        (record_header_t) RECORD_SET_INIT_OPT(0, type, instid, kernid, warpv,
                                              ctaid_serial,
                                              grid,
                                              warpp, sm,
                                              req_size, clock);
      */
      //printf("WRITTEN (%u)\n", alloc_raw);//////////////////////////////
    }


    // write requested addr for each lane
    
    rec_offset = __shfl_sync(active, rec_offset, lowest);
    //if (lane == lowest) {
    
    uint64_t* rec_base = (uint64_t*) &(records[(rec_offset) * RECORD_SIZE +
                                               (rlane_id*24)]);
    rec_base[0] = ctaid_serial;
    rec_base[1] = grid;
    rec_base[2] = addr;
    
    

    // guarantee all writes before to be written to the 'records'
    __threadfence_system();
    
    // commit space in slot, and send full signal to the host
    if (lane == lowest) {
      uint32_t commit_raw = atomicInc(commit, UINT32_MAX) + 1;
      //printf("end (%u / %u)\n", commit_raw, *flushed_v); ////////////////////////
      if (commit_raw == RECORDS_PER_SLOT) {
        //printf("signaled! (%u)\n", RECORDS_PER_SLOT);/////////////////////////////
        //*signal_v = commit_raw; // request flush to host
        *signal_v = RECORDS_PER_SLOT;
        //printf("DEV_commit_v: %u\n", *commit_v);
        //__threadfence_system();
        //*flushed_v = UINT32_MAX; // request sent successfully
      }
    }
    
  }

  

/**************************************************** 
 *  void ___cuprof_trace_ret();
 *
 *  Flush commit_v to signal (host)
 */
  __device__ void ___cuprof_trace_ret(uint32_t* commit, uint32_t* signal,
                                      uint32_t lane) {
    
    uint32_t active;
    asm volatile ("activemask.b32 %0;" : "=r"(active));
    uint32_t lowest = __ffs(active)-1;

    volatile uint32_t* commit_v = commit;

    if (lane == lowest) {
      __threadfence();
      //printf("ret\n");//////////////////
      uint32_t rec_count = *commit_v;

      // if request not sent at the point of return, then send request
      if (rec_count != UINT32_MAX) {
        atomicMax(signal, rec_count); /////////////// need to be fixed
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
