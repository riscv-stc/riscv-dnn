#ifndef _HPM_H
#define _HPM_H

#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <limits.h>
#include <sys/signal.h>

#include "encoding.h"

#define read_csr_safe(reg) ({ register long __tmp asm("a0"); \
            asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
            __tmp; })

volatile uint64_t csr_cycle = 1;
volatile uint64_t csr_opsCommitted = 1;
volatile uint64_t csr_machineClearCycles = 1;
volatile uint64_t csr_icacheStallCycles = 1;
volatile uint64_t csr_branchResteerCycles = 1;
volatile uint64_t csr_defetchLatencyCycles = 1;
volatile uint64_t csr_refetchLatencyCycles = 1;
volatile uint64_t csr_fetchBubblesInsts = 1;
volatile uint64_t csr_renamedInsts = 1;
volatile uint64_t csr_squashCycles = 1;
volatile uint64_t csr_iewExecStallCycle = 1;
volatile uint64_t csr_iewAnyLoadStallCycles = 1;
volatile uint64_t csr_iewStoresStallCycles = 1;
volatile uint64_t csr_branches = 1;
volatile uint64_t csr_vctorVsetvli = 1;
volatile uint64_t csr_vectorVsetvl = 1;
volatile uint64_t csr_vectorVsetivli = 1;
volatile uint64_t csr_branchMispredicts = 1;
volatile uint64_t csr_vectorUnitStrideLoad = 1;
volatile uint64_t csr_vectorUnitStrideStore = 1;
volatile uint64_t csr_vectorStrideLoad = 1;
volatile uint64_t csr_vectorStrideStore = 1;
volatile uint64_t csr_vectorIndexLoad = 1;
volatile uint64_t csr_vectorIndexStore = 1;
volatile uint64_t csr_vectorSegmentLoad = 1;
volatile uint64_t csr_vectorSegmentStore = 1;
volatile uint64_t csr_vectorWholeRegisterLoad = 1;
volatile uint64_t csr_vectorWholeRegisterStore = 1;
volatile uint64_t csr_vectorFloat = 1;
volatile uint64_t csr_vectorInt = 1;
volatile uint64_t csr_floating = 1;
volatile uint64_t csr_scalarLoads = 1;
volatile uint64_t csr_scalarStores = 1;

void enableCount() {
    write_csr(mcounteren, -1); // Enable supervisor use of all perf counters
    write_csr(scounteren, -1); // Enable user use of all perf counters
}

static inline void startCount()
{
    csr_opsCommitted             = read_csr_safe(instret);
    csr_defetchLatencyCycles     = read_csr_safe(hpmcounter6);
    csr_refetchLatencyCycles     = read_csr_safe(hpmcounter7);
    csr_fetchBubblesInsts        = read_csr_safe(hpmcounter8);
    csr_renamedInsts             = read_csr_safe(hpmcounter9);
    csr_squashCycles             = read_csr_safe(hpmcounter10);
    csr_iewExecStallCycle        = read_csr_safe(hpmcounter11);
    csr_iewAnyLoadStallCycles    = read_csr_safe(hpmcounter12);
    csr_iewStoresStallCycles     = read_csr_safe(hpmcounter13);
    csr_branchMispredicts        = read_csr_safe(hpmcounter18);
    csr_machineClearCycles       = read_csr_safe(hpmcounter3);
    csr_cycle                    = read_csr_safe(cycle);

}

static inline void stopCount()
{
    csr_cycle                    = read_csr_safe(cycle)        - csr_cycle;
    csr_opsCommitted             = read_csr_safe(instret)      - csr_opsCommitted;
    csr_machineClearCycles       = read_csr_safe(hpmcounter3)  - csr_machineClearCycles;
    csr_defetchLatencyCycles     = read_csr_safe(hpmcounter6)  - csr_defetchLatencyCycles;
    csr_refetchLatencyCycles     = read_csr_safe(hpmcounter7)  - csr_refetchLatencyCycles;
    csr_fetchBubblesInsts        = read_csr_safe(hpmcounter8)  - csr_fetchBubblesInsts;
    csr_renamedInsts             = read_csr_safe(hpmcounter9)  - csr_renamedInsts;
    csr_squashCycles             = read_csr_safe(hpmcounter10) - csr_squashCycles;
    csr_iewExecStallCycle        = read_csr_safe(hpmcounter11) - csr_iewExecStallCycle;
    csr_iewAnyLoadStallCycles    = read_csr_safe(hpmcounter12) - csr_iewAnyLoadStallCycles;
    csr_iewStoresStallCycles     = read_csr_safe(hpmcounter13) - csr_iewStoresStallCycles;
    csr_branchMispredicts        = read_csr_safe(hpmcounter18) - csr_branchMispredicts;

}

static inline void printfCount()
{
    printf("Perf: %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu\n",
            csr_cycle                   ,
            csr_opsCommitted            ,
            csr_machineClearCycles      ,
            csr_defetchLatencyCycles    ,
            csr_refetchLatencyCycles    ,
            csr_fetchBubblesInsts       ,
            csr_renamedInsts            ,
            csr_squashCycles            ,
            csr_iewExecStallCycle       ,
            csr_iewAnyLoadStallCycles   ,
            csr_iewStoresStallCycles    ,
            csr_branchMispredicts      
            );
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

static inline int insnInfoCntGet(){
  long instret, cycles;
  long intTotal, intLoad, intStore, intAmo, intSystem, intFence, intFencei, intBranch;
  long intJal, intJalr, intAlu, intMul, intDiv;
  long fpTotal, fpLoad, fpStore, fpFpu, fpDiv;
  long rvvTotal, rvvVset, rvvLoad, rvvStore, rvvInt, rvvFloat;

  cycles    = read_csr(mcycle);
  instret   = read_csr(minstret);

  intTotal  = GET_PERCNT(3);
  intLoad   = GET_PERCNT(4);
  intStore  = GET_PERCNT(5);
  intAmo    = GET_PERCNT(6);
  intSystem = GET_PERCNT(7);
  intFence  = GET_PERCNT(8);
  intFencei = GET_PERCNT(9);
  intBranch = GET_PERCNT(10);
  intJal    = GET_PERCNT(11);
  intJalr   = GET_PERCNT(12);
  intAlu    = GET_PERCNT(13);
  intMul    = GET_PERCNT(14);
  intDiv    = GET_PERCNT(15);
  fpTotal   = GET_PERCNT(16);
  fpLoad    = GET_PERCNT(17);
  fpStore   = GET_PERCNT(18);
  fpFpu     = GET_PERCNT(19);
  fpDiv     = GET_PERCNT(20);
  rvvTotal  = GET_PERCNT(21);
  rvvVset   = GET_PERCNT(22);
  rvvLoad   = GET_PERCNT(23);
  rvvStore  = GET_PERCNT(24);
  rvvInt    = GET_PERCNT(25);
  rvvFloat  = GET_PERCNT(26);


  printf("instret:%ld\n", (long)(instret));
  printf("cycles:%ld\n",  (long)(cycles));

  printf("intTotal:%ld\n",  intTotal);
  printf("intLoad:%ld\n",   intLoad);
  printf("intStore:%ld\n",  intStore);
  printf("intAmo:%ld\n",    intAmo);
  printf("intSystem:%ld\n", intSystem);
  printf("intFence:%ld\n",  intFence);
  printf("intFencei:%ld\n", intFencei);
  printf("intBranch:%ld\n", intBranch);
  printf("intJal:%ld\n",    intJal);
  printf("intJalr:%ld\n",   intJalr);
  printf("intAlu:%ld\n",    intAlu);
  printf("intMul:%ld\n",    intMul);
  printf("intDiv:%ld\n",    intDiv);
  printf("fpTotal:%ld\n",   fpTotal);
  printf("fpLoad:%ld\n",    fpLoad);
  printf("fpStore:%ld\n",   fpStore);
  printf("fpFpu:%ld\n",     fpFpu);
  printf("fpDiv:%ld\n",     fpDiv);
  printf("rvvTotal:%ld\n",  rvvTotal);
  printf("rvvVset:%ld\n",   rvvVset);
  printf("rvvLoad:%ld\n",   rvvLoad);
  printf("rvvStore:%ld\n",  rvvStore);
  printf("rvvInt:%ld\n",    rvvInt);
  printf("rvvFloat:%ld\n",  rvvFloat);

  return 0;
}

// top-down hpm counter
static inline int topDownCntSet(){
  write_csr(scounteren, -1);
  write_csr(mcounteren, -1);

  SET_PERFCNT( 3,  8, 0); // slots issued
  SET_PERFCNT( 4,  9, 0); // fetch bubbles
  SET_PERFCNT( 5, 10, 0); // branch instruction retired

  SET_PERFCNT( 6,  8, 1); // bad resteers
  SET_PERFCNT( 7,  9, 1); // recovery bubbles
  SET_PERFCNT( 8, 10, 1); // unknown branch steers
  SET_PERFCNT( 9, 11, 1); // branch miss retired
  SET_PERFCNT(10, 12, 1); // machine clears
  SET_PERFCNT(11, 13, 1); // i-Cache stall
  SET_PERFCNT(12, 14, 1); // i-TLB stall
  SET_PERFCNT(13, 15, 1); // mem stall on any load
  SET_PERFCNT(14, 16, 1); // mem stall on any store

  SET_PERFCNT(15, 16, 2); // Mem Unit valids
  SET_PERFCNT(16, 17, 2); // Jmp Unit valids
  SET_PERFCNT(17, 18, 2); // ALU Unit valids
  SET_PERFCNT(18, 19, 2); // FPU EXE Unit valids
  SET_PERFCNT(19, 20, 2); // Vector EXE Units valids
  SET_PERFCNT(20, 21, 2); // Vector VMX Units valids
  SET_PERFCNT(21, 22, 2); // divider busy cycles
  SET_PERFCNT(22, 11, 2); // execution stalls (few = 1)

  SET_PERFCNT(23, 23, 1); // rob cycles
  SET_PERFCNT(24, 17, 1); // mem stall on L1 miss

  SET_PERFCNT(25, 21, 5); // retired float total
  SET_PERFCNT(26, 25, 5); // retired float div
  SET_PERFCNT(27, 26, 5); // retired vector total
  SET_PERFCNT(28, 27, 5); // retired vector vset
  SET_PERFCNT(29, 28, 5); // retired vector load
  SET_PERFCNT(30, 29, 5); // retired vector store
  SET_PERFCNT(31, 20, 5); // retired int div

  write_csr(minstret, 0);
  write_csr(mcycle, 0);

  return 0;
}

static inline int topDownCntGet(){
  long instret, cycles;
  long slotsIss, fetchBubbles, branchCnt;
  long badResteers, recovery, unknownBranch, branchMiss, machineClear;
  long iCacheStall, iTLBStall, memLoadStall, memStoreStall;
  long memUtil, jmpUtil, aluUtil, fpuUtil, vecUtil, vmxUtil, exeStall;
  long robStall, memStallL1Miss, divBusyCycles;
  long fpTotal, fpDiv, rvvTotal, rvvVset, rvvLoad, rvvStore, intDiv, intTotal;

  instret        = read_csr(minstret);
  cycles         = read_csr(mcycle);
  slotsIss       = GET_PERCNT(3);
  fetchBubbles   = GET_PERCNT(4);
  branchCnt      = GET_PERCNT(5);
  badResteers    = GET_PERCNT(6);
  recovery       = GET_PERCNT(7);    
  unknownBranch  = GET_PERCNT(8);
  branchMiss     = GET_PERCNT(9);
  machineClear   = GET_PERCNT(10);
  iCacheStall    = GET_PERCNT(11);
  iTLBStall      = GET_PERCNT(12);
  memLoadStall   = GET_PERCNT(13);
  memStoreStall  = GET_PERCNT(14);
  memUtil        = GET_PERCNT(15);
  jmpUtil        = GET_PERCNT(16);
  aluUtil        = GET_PERCNT(17);
  fpuUtil        = GET_PERCNT(18);
  vecUtil        = GET_PERCNT(19);
  vmxUtil        = GET_PERCNT(20);
  divBusyCycles  = GET_PERCNT(21);
  exeStall       = GET_PERCNT(22);
  robStall       = GET_PERCNT(23);
  memStallL1Miss = GET_PERCNT(24);
  fpTotal        = GET_PERCNT(25);
  fpDiv          = GET_PERCNT(26);
  rvvTotal       = GET_PERCNT(27);
  rvvVset        = GET_PERCNT(28);
  rvvLoad        = GET_PERCNT(29);
  rvvStore       = GET_PERCNT(30);
  intDiv         = GET_PERCNT(31);
  intTotal       = instret - fpTotal - rvvTotal;

  printf("instret:%ld\n", (long)(instret));
  printf("cycles:%ld\n",  (long)(cycles));

  printf("slotsIssed:%ld\n",           slotsIss);
  printf("fetchBubbles:%ld\n",         fetchBubbles);
  printf("branchRetired:%ld\n",        branchCnt);
  printf("badResteers:%ld\n",          badResteers);
  printf("recoveryCycles:%ld\n",       recovery);
  printf("unknowBanchCycles:%ld\n",    unknownBranch);
  printf("brMispredRetired:%ld\n",     branchMiss);
  printf("machineClears:%ld\n",        machineClear);
  printf("iCacheStallCycles:%ld\n",    iCacheStall);
  printf("iTLBStallCycles:%ld\n",      iTLBStall);
  printf("memStallsAnyLoad:%ld\n",     memLoadStall);
  printf("memStallsStores:%ld\n",      memStoreStall);
  printf("memUnitUtilization:%ld\n",   memUtil);
  printf("jmpUnitUtilization:%ld\n",   jmpUtil);
  printf("aluUnitUtilization:%ld\n",   aluUtil);
  printf("fpuUnitUtilization:%ld\n",   fpuUtil);
  printf("vecUnitUtilization:%ld\n",   vecUtil);
  printf("vmxUnitUtilization:%ld\n",   vmxUtil);
  printf("divBusyCycles:%ld\n",        divBusyCycles);
  printf("exeStallCycles:%ld\n",       exeStall);
  printf("robStallCycles:%ld\n",       robStall);
  printf("memStallsL1Miss:%ld\n",      memStallL1Miss);

  printf("fpTotalRetired:%ld\n",       fpTotal);
  printf("fpDividerRetired:%ld\n",     fpDiv);
  printf("rvvTotalRetired:%ld\n",      rvvTotal);
  printf("rvvVsetRetired:%ld\n",       rvvVset);
  printf("rvvLoadRetired:%ld\n",       rvvLoad);
  printf("rvvStoreRetired:%ld\n",      rvvStore);
  printf("intDividerRetired:%ld\n",    intDiv);
  printf("intTotalRetired:%ld\n",      intTotal);

  printf("memLatency:%d\n",            0);
  printf("memStallsL2Miss:%d\n",       0);
  printf("memStallsL3Miss:%d\n",       0);

  return 0;
}

#endif
