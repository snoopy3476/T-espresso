#include "../lib/common.h"

extern "C" {



#define __match_any_sync_macro(active, val, return_mask)  { \
  return_mask = 0; \
  for (int i = 31; i >= 0; i--) { \
    uint32_t cur_mask = (val == __shfl_sync(active, val, i)); \
    return_mask <<= 1; \
    return_mask += cur_mask; \
  } \
}

  

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
                                               uint8_t to_be_traced) {

    if (!to_be_traced)
      return;

    uint64_t clock;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(clock));
    uint32_t active = __activemask();
    //asm volatile ("activemask.b32 %0;" : "=r"(active));
    uint32_t lowest = __ffs(active)-1;

    uint32_t rlane_id = __popc(active << (32 - lane)); ////////////
    uint32_t n_active = __popc(active); //////////////

    volatile uint32_t* alloc_v = alloc;
    volatile uint32_t* commit_v = commit;
    volatile uint32_t* flushed_v = flushed;
    volatile uint32_t* signal_v = signal;

    uint32_t rec_offset;
    uint32_t flushed_cur;




    
    uint32_t msb = (uint32_t)(addr >> 32);
    int is_msb_same = 1;
    //uint64_t msb = addr & 0xFFFFFFFF00000000;
    //int addr_len;
    //__match_all_sync(active, msb, &addr_len);

    // addr delta mask
    
    uint64_t addr_prev = __shfl_up_sync(active, addr, 1);
    uint64_t addr_next = __shfl_down_sync(active, addr, 1);
    uint64_t addr_delta_prev = addr - addr_prev;
    uint64_t addr_delta_next = addr_next - addr;
    int is_delta_changed = (addr_delta_prev != addr_delta_next);
    
    uint32_t subfilter_1 = __ballot_sync(active, is_delta_changed);

    uint32_t is_inactive_prev = active << 1;
    uint32_t is_inactive_next = active >> 1;
    uint32_t is_prev_set = subfilter_1 << 1;
    uint32_t inactive_no_write = is_inactive_next | is_prev_set;
    uint32_t inactive_force_write = ~is_inactive_prev;

    uint32_t subfilter_2 = (subfilter_1 & inactive_no_write) | inactive_force_write;

    uint32_t prev_lanes_write_mask = subfilter_2 << (32-1 - lane);
    uint32_t consec_write = __clz(~prev_lanes_write_mask);
    uint32_t is_write = consec_write & 0x1;
    
    uint32_t filter = __ballot_sync(active, is_write);

    
    uint8_t write_pos = __popc(prev_lanes_write_mask);
    uint8_t write_count = __popc(filter);
    uint64_t record_size = RECORD_SIZE(write_count); //RECORD_SIZE(write_count);
    


    // allocate space in slot
    if (lane == lowest) {

      // get the allocated offset
      uint32_t alloc_raw = atomicAdd(alloc, record_size); //atomicInc(alloc, UINT32_MAX);
      //rec_offset = alloc_raw & (RECORDS_PER_SLOT-1);

      // wait until slot is not full
      do {
        flushed_cur = *flushed_v;
      } while ((rec_offset = alloc_raw - flushed_cur) >= SLOT_SIZE - 32);

      // map alloc to physical buf
      //rec_offset = alloc_raw - flushed_now;

      ////////// WRITE DISTRIBUTION (OFF) //////////

      // write header at lowest lane
      /*
      record_header_t* rec_header =
        (record_header_t*) &(records[rec_offset * RECORD_MAX_SIZE]);

      *rec_header =
        (record_header_t) RECORD_SET_INIT_OPT(0, type, instid, kernid, warpv,
                                              ctaid_serial,
                                              grid,
                                              warpp, sm,
                                              req_size, clock);
      */
      //////////////////////////////////////////////
    }

    //uint64_t msb = addr & 0xFFFFFFFF00000000;
    //int addr_len;
    //__match_all_sync(active, msb, &addr_len);
    

    ////////// WRITE DISTRIBUTION (ON) //////////

    // write header

    //__match_all_sync(active, msb, &pred);
    

    uint64_t header_info[RECORD_HEADER_UNIT];
    header_info[0] = RECORD_SET_INIT_IDX_0(0, instid, kernid, warpv);
    header_info[1] = RECORD_SET_INIT_IDX_1(ctaid_serial);
    header_info[2] = RECORD_SET_INIT_IDX_2(grid);
    header_info[3] = RECORD_SET_INIT_IDX_3(warpp, sm);
    header_info[4] = RECORD_SET_INIT_IDX_4(clock);
    header_info[5] = is_msb_same ? RECORD_SET_INIT_IDX_5(addr, active) : 0;

    rec_offset = __shfl_sync(active, rec_offset, lowest);

    volatile record_header_t* rec_header =
      (record_header_t*) &(records[rec_offset]);

    for (int i = rlane_id; i < RECORD_HEADER_UNIT; i += n_active) {
      *((uint64_t*)rec_header + i) = header_info[i];
    }

    //////////////////////////////////////////////


    






    /////////////////////////////////////////////


    //uint32_t filter_final = active;

    //uint32_t write_pos = lane;
    //uint32_t is_write = 1;
    
    // write reqeusted addrs for each lane
    if (is_write) {
      volatile uint64_t* rec_addr =
        (uint64_t*) &(records[rec_offset + RECORD_SIZE(write_pos)]);
      *rec_addr = (uint64_t) addr;
    }



    // guarantee all writes before to be written to the 'records'
    __threadfence_system();

    // commit space in slot, and send full signal to the host
    if (lane == lowest) {
      //uint32_t write_count = __popc(filter_final);
      //if (write_count != 32)
      //printf("write_count = %u\n%x\n%x\n%u\n%u\n%u\n\n%x\n%x\n%x)\n", write_count, addr_prev, addr_next, addr_delta_prev, addr_delta_next, is_delta_changed, subfilter_1, subfilter_2, filter);//////////////////
      uint32_t commit_raw = atomicAdd(commit, record_size) + record_size; //atomicInc(commit, UINT32_MAX) + 1;

      /*
      printf("\n\n\n"
             "addr_prev:        \t%08lX\n"
             "addr_next:        \t%08lX\n"
             "addr_delta_prev:  \t%08lX\n"
             "addr_delta_next:  \t%08lX\n"
             "is_delta_changed: \t%d\n"
             "subfilter_1:      \t%08X\n"
             "is_inactive_prev: \t%08X\n"
             "is_inactive_next: \t%08X\n"
             "is_prev_set:      \t%08X\n"
             "inactive_n_write: \t%08X\n"
             "inactive_f_write: \t%08X\n"
             "subfilter_2:      \t%08X\n"
             "prev_la_w_mask:   \t%u\n"
             "consec_write:     \t%u\n"
             "is_write:         \t%u\n"
             "filter:           \t%08X\n"
             "write_pos:        \t%u\n"
             "write_count:      \t%u\n"
             "record_size:      \t%lu\n"
             ,
           addr_prev, addr_next, addr_delta_prev, addr_delta_next, is_delta_changed, subfilter_1, is_inactive_prev, is_inactive_next, is_prev_set, inactive_no_write, inactive_force_write, subfilter_2, prev_lanes_write_mask, consec_write, is_write, filter, write_pos, write_count, record_size);
      */
      
      if (commit_raw - flushed_cur >= SLOT_SIZE - 32) { //(commit_raw & ((RECORDS_PER_SLOT-1))) == 0) {
        *signal_v = commit_raw; // request flush to host
        //__threadfence_system();
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
