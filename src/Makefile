CLANG := clang
SPIKE := spike
PK := pk

objects = test.cpp mat.cpp

target = rvv-test

$(target) : $(objects)
	$(CLANG)  --target=riscv64-unknown-elf -menable-experimental-extensions -march=rv64gcv1p0 --sysroot=$(SYSROOT_DIR) --gcc-toolchain=$(GCC_TOOLCHAIN_DIR) -o $(target) $(objects)

run :
	$(SPIKE) --isa=rv64gcv $(PK) $(target)

clean :
	rm -f $(target)