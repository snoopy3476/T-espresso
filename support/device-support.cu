#include "../lib/Common.h"

extern "C"
__device__ void __mem_trace (uint8_t* records, uint8_t* allocs, uint8_t* commits,
			     uint64_t desc, uint64_t addr, uint64_t clock, uint32_t slot) {

    
    //uint32_t clock_val;
    //asm volatile ("mov.u32 %0, %%clock;" : "=r"(clock_val));


	
    uint64_t cta = blockIdx.x;
    cta <<= 16;
    cta |= blockIdx.y;
    cta <<= 16;
    cta |= blockIdx.z;

    uint32_t lane_id;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    uint32_t active = __ballot(1);
    uint32_t rlane_id = __popc(active << (32 - lane_id));
    uint32_t n_active = __popc(active);
    uint32_t lowest   = __ffs(active)-1;

    uint32_t *alloc = (uint32_t*)(&allocs[slot * CACHELINE]);
    uint32_t *commit = (uint32_t*)(&commits[slot * CACHELINE]);

    volatile uint32_t *valloc = alloc;
    volatile uint32_t *vcommit = commit;
    unsigned int id = 0;

    if (lane_id == lowest) {
      while(*valloc > (SLOTS_SIZE - 32) || (id = atomicAdd(alloc, n_active)) > (SLOTS_SIZE - 32)) {
        (void)0;
      }
    }

    uint32_t slot_offset = slot * SLOTS_SIZE;
    uint32_t record_offset = __shfl_sync(0xFFFFFFFF, id, lowest) + rlane_id;
    record_t *record = (record_t*) &(records[(slot_offset + record_offset) * RECORD_SIZE]);
    uint32_t warp_id = (threadIdx.x +
                        threadIdx.y * blockDim.x +
                        threadIdx.z * blockDim.x * blockDim.y) / 32;
    //asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warp_id));
    //if (addr == 0)
//	record->desc = 0;
    //  else
    *record = RECORD_SET_INIT(1, (desc >> 28) & 0x0F, (desc >> 32) & 0xFF, warp_id,
                              blockIdx.x, blockIdx.y, blockIdx.z, clock, desc & 0x0FFFFFFF);
    RECORD_ADDR(record, 0) = addr;
    RECORD_ADDR_META(record, 0) = 1;
    /*
    record->header = ((uint32_t)1 << 24) | ((uint32_t)((desc >> 32) | 0xFF) << 16);
    record->warp = warp_id;
    record->cta  = cta;
    record->type_size = (uint32_t)(desc & 0xFFFFFFFF);
    record->clock = clock_val;
    record->addr = addr;
    */
    //record->desc = desc;
    /*
    uint64_t meta = clock_val & 0xFFFFFFFF;
    meta <<= 16;
    meta |= warp_id & 0xFFFF;
    meta <<= 8;
    meta |= lane_id & 0xFF;
    meta <<= 8;
    record->meta = meta;
    */
    __threadfence_system();

    if (lane_id == lowest ) atomicAdd(commit, n_active);
}
