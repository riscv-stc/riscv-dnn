SHELL=/bin/bash

top_dir = ../../..
inc_dir = $(top_dir)/include
src_dir = $(top_dir)/src
includes = -I$(inc_dir)/env -I$(inc_dir)/common -I$(src_dir)
defines = $(DEFS)

# simulators
# supported: spike gem5 vcs
SIM ?= spike
# toolchains
# supported: llvm gnu
TC ?= llvm

SPIKE := spike
SPIKE_ARGS :=

GEM5 ?= $(top_dir)/../gem5
GEM5_OPTS :=
GEM5_ARGS :=

SIMV ?= $(top_dir)/../chipyard/sims/vcs/simv-chipyard-StcBoomConfig-debug
SIMV_ARGS := +fsdbfile=test.fsdb
SIMV_POST := 

PK := pk

ifeq (x$(SIM), xspike)
	SIM_CMD ?= \
		$(SPIKE) --isa=rv64gcv_zfh --varch=vlen:1024,elen:64,slen:1024,mlen:65536 \
			+signature=spike.sig +signature-granularity=32 
	defines += -D__SPIKE__
else ifeq (x$(SIM), xgem5)
	SIM_CMD ?= $(top_dir)/scripts/gem5.sh \
		$(GEM5)/build/RISCV/gem5.opt --listener-mode=off $(GEM5_OPTS) \
		   $(GEM5)/configs/example/fs.py --signature=gem5.sig \
		   --cpu-type=StcBoom --bp-type=LTAGE --num-cpu=1 \
		   --mem-channels=1 --mem-size=3072MB \
		   --caches --cacheline_size=128 \
		   --l1d_size=64kB --l1i_size=64kB --l1i_assoc=8 --l1d_assoc=8 \
		   --l2cache --l2_size=64MB \
		   $(GEM5_ARGS) --kernel=
	SIMV_POST := | tee gem5.log		   
	defines += -D__GEM5__
else ifeq (x$(SIM), xvcs)
	SIM_CMD ?= $(top_dir)/scripts/vcs.sh $(SIMV) +signature=vcs.sig +signature-granularity=32 +permissive +loadmem=test.hex +loadmem_addr=80000000 $(SIMV_ARGS) +permissive-off 
	SIMV_POST := </dev/null 2> >(spike-dasm > vcs.out) | tee vcs.log
endif


# toolchain

PREFIX ?= riscv64-unknown-elf-
OBJDUMP := $(PREFIX)objdump

ifneq (x$(TC), xllvm)
	CC := $(PREFIX)gcc
	LINK := $(PREFIX)gcc
	CFLAGS := -g -march=rv64gv0p10zfh0p1  -DPREALLOCATE=1 -mcmodel=medany -static -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns $(includes) $(defines)
else
	CC := clang --target=riscv64-unknown-elf  -march=rv64gv0p10zfh0p1 -menable-experimental-extensions
	LINK := clang --target=riscv64-unknown-elf  -march=rv64gv0p10zfh0p1 -menable-experimental-extensions
	CFLAGS := -g -mcmodel=medany -mllvm -ffast-math -fno-common -fno-builtin-printf $(includes) $(defines)
endif

LDFLAGS :=-static -nostdlib  -nostartfiles  -T $(inc_dir)/common/test.ld

target_elf = test.elf
target_dump = test.dump
target_map = test.map

objects = test.o crt.o syscalls.o


all: $(target_elf)

syscalls.o: $(inc_dir)/common/syscalls.c
	$(CC) $(CFLAGS) -c -o $@ $<

crt.o: $(inc_dir)/common/crt.S
	$(CC) $(CFLAGS) -c -o $@ $<

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<
	
$(target_elf): $(objects)
	$(LINK) -o $(target_elf) $^ $(LDFLAGS)

$(target_map): $(target_elf)
	$(PREFIX)readelf -s -W $< > $@

run: $(target_elf) $(target_map)
	$(SIM_CMD)$(target_elf) $(SIMV_POST)

dump: $(target_dump)

$(target_dump): $(target_elf)
	$(OBJDUMP) -S $(target_elf) > $(target_dump)
	
clean:
	rm -f $(target_elf) $(objects) $(target_dump) $(target_map) *.sig

