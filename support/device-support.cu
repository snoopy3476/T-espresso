#include "../lib/common.h"

extern "C" {


  
  
#define DEBUG_PRINT {                                                   \
    char str_filter_raw[36] = {};                                       \
    char str_filter_raw_prev_lanes[36] = {};                            \
    char str_filter[36] = {};                                           \
    char str_filter_prev_lanes[36] = {};                                \
                                                                        \
    uint32_t* uint_ptr;                                                 \
    char* str_ptr;                                                      \
                                                                        \
    uint32_t* uint_ptr_list[] = {                                       \
      &filter_raw,                                                      \
      &filter_raw_prev_lanes,                                           \
      &filter,                                                          \
      &filter_prev_lanes                                                \
    };                                                                  \
    char* str_ptr_list[] = {                                            \
      str_filter_raw,                                                   \
      str_filter_raw_prev_lanes,                                        \
      str_filter,                                                       \
      str_filter_prev_lanes                                             \
    };                                                                  \
                                                                        \
    for (int i = 0;                                                     \
         i < sizeof(uint_ptr_list) / sizeof(uint_ptr_list[0]);          \
         i++) {                                                         \
      uint_ptr = uint_ptr_list[i];                                      \
      str_ptr = str_ptr_list[i];                                        \
      for (int i = 0; i < 35; i++) {                                    \
        if (i == 8 || i == 17 || i == 26)                               \
          str_ptr[i] = ' ';                                             \
        else                                                            \
          str_ptr[i] = (*uint_ptr & (0x1 << (i - (i / 9)) ) ?'1':'0');  \
      }                                                                 \
    }                                                                   \
                                                                        \
                                                                        \
    printf("\n"                                                         \
           "lane:             \t%u\n"                                   \
           "addr_prev_prev:   \t%08lX\n"                                \
           "addr_prev:        \t%08lX\n"                                \
           "addr_delta_prev:  \t%08lX\n"                                \
           "addr_delta:       \t%08lX\n"                                \
           "is_delta_changed: \t%c\n"                                   \
           "is_prev_inactive: \t%c\n"                                   \
           "is_write_rawf:    \t%c\n"                                   \
           "filter_raw:       \t%s\n"                                   \
           "filter_raw_prevs: \t%s\n"                                   \
           "consec_write:     \t%u\n"                                   \
           "is_write:         \t%c\n"                                   \
           "filter:           \t%s\n"                                   \
           "filter_prevs:     \t%s\n"                                   \
           "write_pos:        \t%u\n"                                   \
           "write_count:      \t%u\n"                                   \
           "record_size:      \t%lu\n\n\n"                              \
           ,                                                            \
           lane,                                                        \
           addr_prev_prev,                                              \
           addr_prev,                                                   \
           addr_delta_prev,                                             \
           addr_delta,                                                  \
           is_delta_changed?'O':'X',                                    \
           is_prev_inactive?'O':'X',                                    \
           is_write_rawf?'O':'X',                                       \
           str_filter_raw,                                              \
           str_filter_raw_prev_lanes,                                   \
           consec_write,                                                \
           is_write?'O':'X',                                            \
           str_filter,                                                  \
           str_filter_prev_lanes,                                       \
           write_pos,                                                   \
           write_count,                                                 \
           record_size                                                  \
      );                                                                \
  }

  
  

/****************************************************
 *  void ___cuprof_trace();
 *
 *  Write trace data and associated info
 *  to the externally allocated areas.
 */
  __device__ __noinline__ void ___cuprof_trace(uint32_t* alloc, uint32_t* commit,
                                               uint32_t* flushed, uint32_t* signal,
                                               uint8_t* records, uint64_t data,
                                               uint64_t grid, uint64_t ctaid_serial,
                                               uint32_t warpv, uint32_t laneid,
                                               uint32_t instid, uint32_t kernid,
                                               uint32_t sm, uint32_t warpp,
                                               uint8_t to_be_traced) {

    if (!to_be_traced)
      return;

    uint64_t clock = clock64();

    volatile uint32_t* flushed_v = flushed;
    volatile uint32_t* signal_v = signal;
    
    uint32_t active = __activemask();
    uint32_t laneid_leader = __ffs(active)-1;
    
    uint64_t data_prev_prev = __shfl_up_sync(active, data, 2);
    uint64_t data_prev = __shfl_up_sync(active, data, 1);
    uint64_t data_next = __shfl_down_sync(active, data, 1);

    uint32_t lanemask = (0x1 << laneid);
    uint32_t lanemask_prevs = lanemask - 1;
    uint32_t laneid_among_active = __popc(active & lanemask_prevs);
    uint32_t active_count = __popc(active);

    uint32_t rec_offset;
    uint32_t flushed_cur;

    kernid = 1;
    if (instid >= RECORD_UNKNOWN)
      instid = 14;

#ifndef CUPROF_ODE_DISABLE

    // check if msb is the same across all active threads
    
    uint64_t msb = data >> 32;
    uint64_t msb_leader = __shfl_sync(active, msb, laneid_leader);
    uint64_t msb_delta = msb - msb_leader;
    uint32_t is_msb_same = !(__ballot_sync(active, msb_delta));
    

    
    
    // calculate warp writemask + record size
    
    uint64_t data_delta_prev = data_prev - data_prev_prev;
    uint64_t data_delta = data - data_prev;
    uint64_t data_delta_next = data_next - data;
    uint32_t is_delta_changed = (data_delta != data_delta_prev);
    uint32_t is_prev_inactive = (~(active << 1) & lanemask);
    uint32_t is_write_rawf = is_delta_changed | is_prev_inactive;
  
    uint32_t writemask_raw = __ballot_sync(active, is_write_rawf);
    
    uint32_t writemask_raw_prev_lanes = writemask_raw << (32-1 - laneid);
    uint32_t consec_write = __clz(~writemask_raw_prev_lanes);
    // if msb is not same, all thread writes
    uint32_t is_write = (consec_write & 0x1) | (!is_msb_same);
  
    uint32_t writemask = __ballot_sync(active, is_write);
    uint8_t write_count = __popc(writemask);
    uint64_t record_size = RECORD_SIZE(write_count);


    

    // get data write position for the current thread
    
    uint32_t writemask_prev_lanes = writemask & lanemask_prevs;
    uint8_t write_pos = __popc(writemask_prev_lanes);



#else

    uint64_t msb = data >> 32;
    uint32_t is_msb_same = 1;
    uint32_t is_write = 1;
    uint32_t writemask = active;
    uint64_t record_size = RECORD_SIZE(active_count);
    uint8_t write_pos = laneid_among_active;
    
#endif

    
    // initialize record header + data

    uint64_t header_info[RECORD_HEADER_UNIT];
    header_info[0] = 1; // ensure first elem to be non-zero
    header_info[1] = RECORD_SET_HEADER_1(active, (is_msb_same ? writemask : 0)); // if msb is not same, writemask == 0
    header_info[2] = RECORD_SET_HEADER_2(ctaid_serial);
    header_info[3] = RECORD_SET_HEADER_3(grid);
    header_info[4] = RECORD_SET_HEADER_4(warpp, sm);
    header_info[5] = RECORD_SET_HEADER_5(msb, clock);

    
#ifndef CUPROF_ODE_DISABLE
    
    if (is_msb_same)
      data = RECORD_SET_DATA(data_delta_next, data);
    
#endif


#ifndef CUPROF_RMSYNC_DISABLE
    
    // data part nonzero mask
    uint64_t nonzero_mask_data = __ballot_sync(active, data);

    uint64_t nonzero_mask_header = LLGT_BIT_MASK(RECORD_HEADER_UNIT);

    // header part zero mask
    for (int i = 0; i < RECORD_HEADER_UNIT; i++) {
      if (header_info[i] == 0) {
        header_info[i] = -1; // set to non-zero, if header is zero
        nonzero_mask_header ^= ((uint64_t)1 << i); // make zero part to bit 0
      }
    }

    // header part + data part
    uint64_t nonzero_mask = nonzero_mask_header |
      ((writemask & nonzero_mask_data) << RECORD_HEADER_UNIT); // append
    
    header_info[0] = RECORD_SET_HEADER_0(nonzero_mask, kernid, instid, warpv);
    if (data == 0) data = -1; // set to non-zero, if thread data is zero
    ///// need to write non-zero if msb is different
#else
    
    header_info[0] = RECORD_SET_HEADER_0(-1, kernid, instid, warpv);

#endif
    

    //DEBUG_PRINT;


    //////////////////////////
    /*
      uint32_t is_write = 1; //consec_write & 0x1;
    
      uint32_t writemask = 0xFFFFFFFF;

    
      uint8_t write_pos = lane; //__popc(prev_lanes_write_mask);
      uint8_t write_count = 32; //__popc(writemask);
      uint64_t record_size = RECORD_SIZE(write_count);
    */


    // allocate space in slot
    if (laneid == laneid_leader) {

      // get the allocated offset
      uint32_t alloc_raw = atomicAdd(alloc, record_size);
      
      rec_offset = alloc_raw % SLOT_SIZE;

      // wait until slot is not full
      do {
        flushed_cur = *flushed_v;
      } while ((alloc_raw - flushed_cur) >= SLOT_SIZE - RECORD_SIZE_MAX);
    }



    rec_offset = __shfl_sync(active, rec_offset, laneid_leader);

#ifndef CUPROF_CRW_DISABLE

    volatile uint64_t* rec_header = (uint64_t*) (records + rec_offset);

    for (int i = laneid_among_active; i < RECORD_HEADER_UNIT; i += active_count) {
      int rec_i = (rec_offset + sizeof(uint64_t)*i) % SLOT_SIZE;
      *(uint64_t*)(records + rec_i) = header_info[i];
    }
    //////////////////////////////////////////////

#else

    
    if (laneid == laneid_leader) {
      for (int i = 0; i < RECORD_HEADER_UNIT; i++) {
        int rec_i = (rec_offset + sizeof(uint64_t)*i) % SLOT_SIZE;
        *(uint64_t*)(records + rec_i) = header_info[i];
      }
    }

#endif






#ifndef CUPROF_CRW_DISABLE

    if (is_write) {
      int rec_i = (rec_offset + RECORD_SIZE(write_pos)) % SLOT_SIZE;
      volatile uint64_t* rec_data = (uint64_t*) (records + rec_i);
      *rec_data = data;
    }

#else

    uint64_t data_warp[32];
    uint32_t data_count = 0;

    for (int i = 0; i < 32; i++) {

      uint32_t mask = (0x1 << i);
      if (mask & writemask) {
        data_warp[data_count] = __shfl_sync(active, data, i);
        if (data_warp[data_count] == 0)
          data_warp[data_count] = -1;
        data_count++;
      }

    }

    if (laneid == laneid_leader) {
      for (int i = 0; i < data_count; i++) {
        
        int rec_i = (rec_offset + RECORD_SIZE(i)) % SLOT_SIZE;
        volatile uint64_t* rec_data = (uint64_t*) (records + rec_i);
        *rec_data = data_warp[i];
      }
        

    }
    
#endif
    


#ifdef CUPROF_RMSYNC_DISABLE

    // guarantee all writes before to be written to the 'records'
    __threadfence_system();
    
#endif

    // commit space in slot, and send full signal to the host
    if (laneid == laneid_leader) {
      //uint32_t write_count = __popc(writemask_final);
      //if (write_count != 32)
      //printf("write_count = %u\n%x\n%x\n%u\n%u\n%u\n\n%x\n%x\n%x)\n", write_count, data_prev, data_next, data_delta_prev, data_delta_next, is_delta_changed, subwritemask_1, subwritemask_2, writemask);//////////////////
      uint32_t commit_raw = atomicAdd(commit, record_size) + record_size; //atomicInc(commit, UINT32_MAX) + 1;
      
      

#ifndef CUPROF_MULTI_BUF_DISABLE
      uint32_t flush_unit = UNIT_SLOT_SIZE - RECORD_SIZE_MAX;
      uint32_t flush_threshold = UNIT_SLOT_SIZE - (2*RECORD_SIZE_MAX);
#else
      uint32_t flush_unit = SLOT_SIZE - RECORD_SIZE_MAX;
      uint32_t flush_threshold = SLOT_SIZE - (2*RECORD_SIZE_MAX);
#endif
      if (
        ( ((commit_raw - flushed_cur - record_size) % flush_unit) >= flush_threshold )
        &&
        ( (commit_raw - flushed_cur) % flush_unit < flush_threshold )
        ) {
        *signal_v = commit_raw; // request flush to host
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
/*
  return;

  uint32_t active;
  asm volatile ("activemask.b32 %0;" : "=r"(active));
  uint32_t lowest = __ffs(active)-1;

  volatile uint32_t* commit_v = commit;

  if (lane == lowest) {
  //__threadfence();
  //printf("ret\n");//////////////////
  //uint32_t rec_count = *commit_v;

  // if request not sent at the point of return, then send request
  //if (rec_count != UINT32_MAX) {
      
  //atomicMax(signal, *commit_v); /////////////// need to be fixed
  // Bug possibility: what if signal overflows?
      
  //}
  // guarantee write before return
  //__threadfence_system();
  }
*/
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
