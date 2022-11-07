typedef struct {
    long instret, cycles;
    long slotsIssed, fetchBubbles, branchRetired;
    long badResteers, recoveryCycles, unknowBanchCycles, brMispredRetired, machineClears;
    long iCacheStallCycles, iTLBStallCycles, memStallsAnyLoad, memStallsStores;
    long aluUnitUtilization, fpuUnitUtilization, vecUnitUtilization, matUnitUtilization, exeStallCycles;
    long memStallsL1Miss, divBusyCycles;
    long fpTotalRetired, fpDividerRetired, intDividerRetired, intTotalRetired;
    long rvvTotalRetired, rvvLoadRetired, rvvStoreRetired;
    long rvmTotalRetired, rvmMsetRetired, rvmLoadRetired, rvmStoreRetired;

    long branchResteerCycles;
    long refetchLatencyCycles, defetchLatencyCycles, branches;
    long renamedInsts, squashCycles;
    long vectorVsetvli, vectorVsetvl, vectorVsetivli;
    long vectorUnitStrideLoad, vectorUnitStrideStore, vectorStirdeLoad, vectorStrideStore, vectorIndexLoad, vectorIndexStore;
    long vectorSegmentLoad, vectorSegmentStore, vectorWholeRegisterLoad, vectorWholeRegisterStore;
    long vectorFloat, vectorInt;
    long scalar_loads, scalar_stores;
} tma_data_t;

typedef struct {
    long instret, cycles;
    long intTotalRetired, intLoad, intStore, intAmo, intSystem, intFence, intFencei, branchRetired;
    long intJal, intJalr, intAlu, intMul, intDividerRetired;
    long fpTotalRetired, fpLoad, fpStore, fpFpu, fpDividerRetired;
    long rvvTotalRetired, rvvVset, rvvLoadRetired, rvvStoreRetired, rvvInt, rvvFloat;
} instuction_data_t;