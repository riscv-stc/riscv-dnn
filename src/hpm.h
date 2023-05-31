#ifndef _HPM_H
#define _HPM_H

#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <limits.h>
#include <sys/signal.h>

#include "encoding.h"

#include "perf-data.h"

#define CORE_MAX 8

#define read_csr_safe(reg) ({ register long __tmp asm("a0"); \
            asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
            __tmp; })

static tma_data_t perf_info[CORE_MAX] __attribute__((__section__(".pfdata.output")));

void enableCount() {
    write_csr(mcounteren, -1); // Enable supervisor use of all perf counters
    write_csr(scounteren, -1); // Enable user use of all perf counters
}


static inline void startCount()
{
    int core_id = read_csr(mhartid);
    perf_info[core_id].instret                  = read_csr_safe(instret);
    perf_info[core_id].machineClears            = read_csr_safe(hpmcounter3);
    perf_info[core_id].iCacheStallCycles        = read_csr_safe(hpmcounter4);
    perf_info[core_id].branchResteerCycles      = read_csr_safe(hpmcounter5);
    perf_info[core_id].defetchLatencyCycles     = read_csr_safe(hpmcounter6);
    perf_info[core_id].refetchLatencyCycles     = read_csr_safe(hpmcounter7);
    perf_info[core_id].fetchBubbles             = read_csr_safe(hpmcounter8);
    perf_info[core_id].renamedInsts             = read_csr_safe(hpmcounter9);
    perf_info[core_id].squashCycles             = read_csr_safe(hpmcounter10);
    perf_info[core_id].exeStallCycles           = read_csr_safe(hpmcounter11);
    perf_info[core_id].memStallsAnyLoad         = read_csr_safe(hpmcounter12);
    perf_info[core_id].memStallsStores          = read_csr_safe(hpmcounter13);
    perf_info[core_id].branches                 = read_csr_safe(hpmcounter14);
    perf_info[core_id].vectorVsetvli            = read_csr_safe(hpmcounter15);
    perf_info[core_id].vectorVsetvl             = read_csr_safe(hpmcounter16);
    perf_info[core_id].vectorVsetivli           = read_csr_safe(hpmcounter17);
    perf_info[core_id].brMispredRetired         = read_csr_safe(hpmcounter18);
    perf_info[core_id].vectorUnitStrideLoad     = read_csr_safe(hpmcounter19);
    perf_info[core_id].vectorUnitStrideStore    = read_csr_safe(hpmcounter20);
    perf_info[core_id].vectorStirdeLoad         = read_csr_safe(hpmcounter21);
    perf_info[core_id].vectorStrideStore        = read_csr_safe(hpmcounter22);
    perf_info[core_id].vectorIndexLoad          = read_csr_safe(hpmcounter23);
    perf_info[core_id].vectorIndexStore         = read_csr_safe(hpmcounter24);
    perf_info[core_id].vectorSegmentLoad        = read_csr_safe(hpmcounter25);
    perf_info[core_id].vectorSegmentStore       = read_csr_safe(hpmcounter26);
    perf_info[core_id].vectorWholeRegisterLoad  = read_csr_safe(hpmcounter27);
    perf_info[core_id].vectorWholeRegisterStore = read_csr_safe(hpmcounter28);
    perf_info[core_id].vectorFloat              = read_csr_safe(hpmcounter29);
    perf_info[core_id].vectorInt                = read_csr_safe(hpmcounter30);
    perf_info[core_id].cycles                   = read_csr_safe(cycle);

}

static inline void stopCount()
{
    int core_id = read_csr(mhartid);
    perf_info[core_id].cycles                   = read_csr_safe(cycle)         - perf_info[core_id].cycles                  ;
    perf_info[core_id].instret                  = read_csr_safe(instret)       - perf_info[core_id].instret                 ;
    perf_info[core_id].machineClears            = read_csr_safe(hpmcounter3)   - perf_info[core_id].machineClears           ;
    perf_info[core_id].iCacheStallCycles        = read_csr_safe(hpmcounter4)   - perf_info[core_id].iCacheStallCycles       ;
    perf_info[core_id].branchResteerCycles      = read_csr_safe(hpmcounter5)   - perf_info[core_id].branchResteerCycles     ;
    perf_info[core_id].defetchLatencyCycles     = read_csr_safe(hpmcounter6)   - perf_info[core_id].defetchLatencyCycles    ;
    perf_info[core_id].refetchLatencyCycles     = read_csr_safe(hpmcounter7)   - perf_info[core_id].refetchLatencyCycles    ;
    perf_info[core_id].fetchBubbles             = read_csr_safe(hpmcounter8)   - perf_info[core_id].fetchBubbles            ;
    perf_info[core_id].renamedInsts             = read_csr_safe(hpmcounter9)   - perf_info[core_id].renamedInsts            ;
    perf_info[core_id].squashCycles             = read_csr_safe(hpmcounter10)  - perf_info[core_id].squashCycles            ;
    perf_info[core_id].exeStallCycles           = read_csr_safe(hpmcounter11)  - perf_info[core_id].exeStallCycles          ;
    perf_info[core_id].memStallsAnyLoad         = read_csr_safe(hpmcounter12)  - perf_info[core_id].memStallsAnyLoad        ;
    perf_info[core_id].memStallsStores          = read_csr_safe(hpmcounter13)  - perf_info[core_id].memStallsStores         ;
    perf_info[core_id].branches                 = read_csr_safe(hpmcounter14)  - perf_info[core_id].branches                ;
    perf_info[core_id].vectorVsetvli            = read_csr_safe(hpmcounter15)  - perf_info[core_id].vectorVsetvli           ;
    perf_info[core_id].vectorVsetvl             = read_csr_safe(hpmcounter16)  - perf_info[core_id].vectorVsetvl            ;
    perf_info[core_id].vectorVsetivli           = read_csr_safe(hpmcounter17)  - perf_info[core_id].vectorVsetivli          ;
    perf_info[core_id].brMispredRetired         = read_csr_safe(hpmcounter18)  - perf_info[core_id].brMispredRetired        ;
    perf_info[core_id].vectorUnitStrideLoad     = read_csr_safe(hpmcounter19)  - perf_info[core_id].vectorUnitStrideLoad    ;
    perf_info[core_id].vectorUnitStrideStore    = read_csr_safe(hpmcounter20)  - perf_info[core_id].vectorUnitStrideStore   ;
    perf_info[core_id].vectorStirdeLoad         = read_csr_safe(hpmcounter21)  - perf_info[core_id].vectorStirdeLoad        ;
    perf_info[core_id].vectorStrideStore        = read_csr_safe(hpmcounter22)  - perf_info[core_id].vectorStrideStore       ;
    perf_info[core_id].vectorIndexLoad          = read_csr_safe(hpmcounter23)  - perf_info[core_id].vectorIndexLoad         ;
    perf_info[core_id].vectorIndexStore         = read_csr_safe(hpmcounter24)  - perf_info[core_id].vectorIndexStore        ;
    perf_info[core_id].vectorSegmentLoad        = read_csr_safe(hpmcounter25)  - perf_info[core_id].vectorSegmentLoad       ;
    perf_info[core_id].vectorSegmentStore       = read_csr_safe(hpmcounter26)  - perf_info[core_id].vectorSegmentStore      ;
    perf_info[core_id].vectorWholeRegisterLoad  = read_csr_safe(hpmcounter27)  - perf_info[core_id].vectorWholeRegisterLoad ;
    perf_info[core_id].vectorWholeRegisterStore = read_csr_safe(hpmcounter28)  - perf_info[core_id].vectorWholeRegisterStore;
    perf_info[core_id].vectorFloat              = read_csr_safe(hpmcounter29)  - perf_info[core_id].vectorFloat             ;
    perf_info[core_id].vectorInt                = read_csr_safe(hpmcounter30)  - perf_info[core_id].vectorInt               ;
    

}


#define SET_PERFCNT(mhpmcnt, eventid, enventclass) \
  write_csr(mhpmcounter ## mhpmcnt, 0); \
  write_csr(mhpmevent ## mhpmcnt, HPM_EVENTID_ ## eventid|HPM_EVENTCLASS_ ## enventclass);

#define SHOW_PERFCNT(fmt, mhpmcnt) \
  printf(fmt, (long)(read_csr(mhpmcounter ## mhpmcnt)));

#define GET_PERCNT(mhpmcnt) \
  read_csr(mhpmcounter ## mhpmcnt);

/*! @brief Macros for valid Event IDs */
#define HPM_EVENTID_8  (1UL << 8)
#define HPM_EVENTID_9  (1UL << 9)
#define HPM_EVENTID_10 (1UL << 10)
#define HPM_EVENTID_11 (1UL << 11)
#define HPM_EVENTID_12 (1UL << 12)
#define HPM_EVENTID_13 (1UL << 13)
#define HPM_EVENTID_14 (1UL << 14)
#define HPM_EVENTID_15 (1UL << 15)
#define HPM_EVENTID_16 (1UL << 16)
#define HPM_EVENTID_17 (1UL << 17)
#define HPM_EVENTID_18 (1UL << 18)
#define HPM_EVENTID_19 (1UL << 19)
#define HPM_EVENTID_20 (1UL << 20)
#define HPM_EVENTID_21 (1UL << 21)
#define HPM_EVENTID_22 (1UL << 22)
#define HPM_EVENTID_23 (1UL << 23)
#define HPM_EVENTID_24 (1UL << 24)
#define HPM_EVENTID_25 (1UL << 25)
#define HPM_EVENTID_26 (1UL << 26)
#define HPM_EVENTID_27 (1UL << 27)
#define HPM_EVENTID_28 (1UL << 28)
#define HPM_EVENTID_29 (1UL << 29)
#define HPM_EVENTID_30 (1UL << 30)
#define HPM_EVENTID_31 (1UL << 31)
#define HPM_EVENTID_32 (1UL << 32)
#define HPM_EVENTID_33 (1UL << 33)
#define HPM_EVENTID_34 (1UL << 34)
#define HPM_EVENTID_35 (1UL << 35)
#define HPM_EVENTID_36 (1UL << 36)
#define HPM_EVENTID_37 (1UL << 37)
#define HPM_EVENTID_38 (1UL << 38)
#define HPM_EVENTID_39 (1UL << 39)
#define HPM_EVENTID_40 (1UL << 40)

/*! @brief Macros for valid Event Class */
#define HPM_EVENTCLASS_0 (0UL)
#define HPM_EVENTCLASS_1 (1UL)
#define HPM_EVENTCLASS_2 (2UL)
#define HPM_EVENTCLASS_3 (3UL)
#define HPM_EVENTCLASS_4 (4UL)
#define HPM_EVENTCLASS_5 (5UL)
#define HPM_EVENTCLASS_6 (6UL)
#define HPM_EVENTCLASS_7 (7UL)
#define HPM_EVENTCLASS_8 (8UL)

// info hpm
static inline int insnInfoCntSet(){
  write_csr(scounteren, -1);
  write_csr(mcounteren, -1);

  SET_PERFCNT(16, 21, 5); // fp total
  SET_PERFCNT(17, 22, 5); // fp load
  SET_PERFCNT(18, 23, 5); // fp store
  SET_PERFCNT(19, 24, 5); // fp fpu
  SET_PERFCNT(20, 25, 5); // fp div
  SET_PERFCNT(21, 26, 5); // vector total
  SET_PERFCNT(22, 27, 5); // vector vset
  SET_PERFCNT(23, 28, 5); // vector load
  SET_PERFCNT(24, 29, 5); // vector store
  SET_PERFCNT(25, 30, 5); // vector int
  SET_PERFCNT(26, 31, 5); // vector float
  SET_PERFCNT( 6, 11, 5); // int amo
  SET_PERFCNT( 8, 13, 5); // int fence
  SET_PERFCNT( 9, 14, 5); // int fencei
  SET_PERFCNT(10, 15, 5); // int branch
  SET_PERFCNT(11, 16, 5); // int jal
  SET_PERFCNT(12, 17, 5); // int jalr
  SET_PERFCNT(13, 18, 5); // int alu
  SET_PERFCNT(14, 19, 5); // int mul
  SET_PERFCNT(15, 20, 5); // int div
  SET_PERFCNT( 4,  9, 5); // int load
  SET_PERFCNT( 5, 10, 5); // int store
  SET_PERFCNT( 3,  8, 5); // int total
  SET_PERFCNT( 7, 12, 5); // int system

  write_csr(minstret, 0);
  write_csr(mcycle, 0);

  return 0;
}

static instuction_data_t insn_info[CORE_MAX] __attribute__((__section__(".pfdata.output")));

static inline int insnInfoCntGet(){
  int core_id = read_csr(mhartid);

  insn_info[core_id].cycles    = read_csr(mcycle);
  insn_info[core_id].instret   = read_csr(minstret);

  insn_info[core_id].intTotalRetired  = GET_PERCNT(3);
  insn_info[core_id].intLoad   = GET_PERCNT(4);
  insn_info[core_id].intStore  = GET_PERCNT(5);
  insn_info[core_id].intAmo    = GET_PERCNT(6);
  insn_info[core_id].intSystem = GET_PERCNT(7);
  insn_info[core_id].intFence  = GET_PERCNT(8);
  insn_info[core_id].intFencei = GET_PERCNT(9);
  insn_info[core_id].branchRetired = GET_PERCNT(10);
  insn_info[core_id].intJal    = GET_PERCNT(11);
  insn_info[core_id].intJalr   = GET_PERCNT(12);
  insn_info[core_id].intAlu    = GET_PERCNT(13);
  insn_info[core_id].intMul    = GET_PERCNT(14);
  insn_info[core_id].intDividerRetired    = GET_PERCNT(15);
  insn_info[core_id].fpTotalRetired    = GET_PERCNT(16);
  insn_info[core_id].fpLoad    = GET_PERCNT(17);
  insn_info[core_id].fpStore   = GET_PERCNT(18);
  insn_info[core_id].fpFpu     = GET_PERCNT(19);
  insn_info[core_id].fpDividerRetired     = GET_PERCNT(20);
  insn_info[core_id].rvvTotalRetired  = GET_PERCNT(21);
  insn_info[core_id].rvvVset   = GET_PERCNT(22);
  insn_info[core_id].rvvLoadRetired   = GET_PERCNT(23);
  insn_info[core_id].rvvStoreRetired  = GET_PERCNT(24);
  insn_info[core_id].rvvInt    = GET_PERCNT(25);
  insn_info[core_id].rvvFloat  = GET_PERCNT(26);

  return 0;
}

// top-down hpm counter
static inline int topDownCntSet(){
  write_csr(scounteren, -1);
  write_csr(mcounteren, -1);

  SET_PERFCNT( 6,  8, 1); // bad resteers
  SET_PERFCNT( 7,  9, 1); // recovery bubbles
  SET_PERFCNT( 8, 10, 1); // unknown branch steers
  SET_PERFCNT( 9, 11, 1); // branch miss retired
  SET_PERFCNT(10, 12, 1); // machine clears
  SET_PERFCNT(11, 13, 1); // i-Cache stall
  SET_PERFCNT(12, 14, 1); // i-TLB stall
  SET_PERFCNT(13, 15, 1); // mem stall on any load
  SET_PERFCNT(14, 16, 1); // mem stall on any store
  SET_PERFCNT(15, 17, 1); // mem stall on L1 miss

  SET_PERFCNT(16, 18, 2); // ALU Unit valids
  SET_PERFCNT(17, 19, 2); // FPU EXE Unit valids
  SET_PERFCNT(18, 20, 2); // Vector EXE Units valids
  SET_PERFCNT(19, 21, 2); // Matrix EXE Units valids
  SET_PERFCNT(20, 22, 2); // divider busy cycles
  SET_PERFCNT(21, 11, 2); // execution stalls (few = 1)

  SET_PERFCNT(22, 20, 5); // retired int div
  SET_PERFCNT(23, 21, 5); // retired float  total
  SET_PERFCNT(24, 25, 5); // retired float  div
  SET_PERFCNT(25, 26, 5); // retired vector total
  SET_PERFCNT(26, 28, 5); // retired vector load
  SET_PERFCNT(27, 29, 5); // retired vector store
  SET_PERFCNT(28, 32, 5); // retired matrix total
  SET_PERFCNT(29, 33, 5); // retired matrix set
  SET_PERFCNT(30, 34, 5); // retired matrix load
  SET_PERFCNT(31, 35, 5); // retired matrix store

  SET_PERFCNT( 5, 10, 0); // branch instruction retired
  SET_PERFCNT( 3,  8, 0); // slots issued
  SET_PERFCNT( 4,  9, 0); // fetch bubbles

  write_csr(minstret, 0);
  write_csr(mcycle, 0);

  return 0;
}



static inline int topDownCntGet(){
  int core_id = read_csr(mhartid);

  perf_info[core_id].instret        = read_csr(minstret);
  perf_info[core_id].cycles         = read_csr(mcycle);
  perf_info[core_id].slotsIssed           = GET_PERCNT(3);
  perf_info[core_id].fetchBubbles   = GET_PERCNT(4);
  perf_info[core_id].branchRetired      = GET_PERCNT(5);
  perf_info[core_id].badResteers    = GET_PERCNT(6);
  perf_info[core_id].recoveryCycles       = GET_PERCNT(7);    
  perf_info[core_id].unknowBanchCycles    = GET_PERCNT(8);
  perf_info[core_id].brMispredRetired     = GET_PERCNT(9);
  perf_info[core_id].machineClears        = GET_PERCNT(10);
  perf_info[core_id].iCacheStallCycles    = GET_PERCNT(11);
  perf_info[core_id].iTLBStallCycles      = GET_PERCNT(12);
  perf_info[core_id].memStallsAnyLoad     = GET_PERCNT(13);
  perf_info[core_id].memStallsStores      = GET_PERCNT(14);
  perf_info[core_id].memStallsL1Miss = GET_PERCNT(15);
  perf_info[core_id].aluUnitUtilization        = GET_PERCNT(16);
  perf_info[core_id].fpuUnitUtilization        = GET_PERCNT(17);
  perf_info[core_id].vecUnitUtilization        = GET_PERCNT(18);
  perf_info[core_id].matUnitUtilization        = GET_PERCNT(19);
  perf_info[core_id].divBusyCycles  = GET_PERCNT(20);
  perf_info[core_id].exeStallCycles       = GET_PERCNT(21);
  perf_info[core_id].intDividerRetired         = GET_PERCNT(22);
  perf_info[core_id].fpTotalRetired         = GET_PERCNT(23);
  perf_info[core_id].fpDividerRetired          = GET_PERCNT(24);
  perf_info[core_id].rvvTotalRetired       = GET_PERCNT(25);
  perf_info[core_id].rvvLoadRetired        = GET_PERCNT(26);
  perf_info[core_id].rvvStoreRetired       = GET_PERCNT(27);
  perf_info[core_id].rvmTotalRetired       = GET_PERCNT(28);
  perf_info[core_id].rvmMsetRetired        = GET_PERCNT(29);
  perf_info[core_id].rvmLoadRetired        = GET_PERCNT(30);
  perf_info[core_id].rvmStoreRetired       = GET_PERCNT(31);
  perf_info[core_id].intTotalRetired       = perf_info[core_id].instret - perf_info[core_id].fpTotalRetired  - perf_info[core_id].rvvTotalRetired - perf_info[core_id].rvmTotalRetired;

  return 0;
}

#endif
