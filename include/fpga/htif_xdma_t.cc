#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <arpa/inet.h>
#include "htif_xdma_t.h"

#define NHARTS_MAX 16
#define MSIP_BASE 0x2000000
#define XDMA_READ_DEVICE "/dev/xdma0_c2h_0"
#define XDMA_WRIT_DEVICE "/dev/xdma0_h2c_0"

#define XDMA_READ_DEVICE1 "/dev/xdma0_c2h_1"
#define XDMA_REG_WR "/dev/xdma0_user"

#define CHIP_RESET_REG_ADDR           0x110000
#define MEMALIGN_SIZE             0x1000
#define RW_MAX_SIZE               0x7ffff000
#define MCU_RESET_OVER_ADDR     0x2011000
#define PROGRAM_RUN_OVER_ADDR_VAL 0x3344

#if __BYTE_ORDER == __LITTLE_ENDIAN
#define ltohl(x)       (x)
#define ltohs(x)       (x)
#define htoll(x)       (x)
#define htols(x)       (x)
#elif __BYTE_ORDER == __BIG_ENDIAN
#define ltohl(x)     __bswap_32(x)
#define ltohs(x)     __bswap_16(x)
#define htoll(x)     __bswap_32(x)
#define htols(x)     __bswap_16(x)
#endif

void htif_xdma_t::open_dev(){
  pcie_mem_rd_fd = open(XDMA_READ_DEVICE, O_RDWR | O_SYNC);
  pcie_mem_wd_fd = open(XDMA_WRIT_DEVICE, O_RDWR | O_SYNC);
  pcie_reg_wr_fd = open(XDMA_REG_WR, O_RDWR | O_SYNC);
  pcie_mem_rd_fd1 = open(XDMA_READ_DEVICE1, O_RDWR | O_SYNC);
  if (pcie_mem_rd_fd < 0 || pcie_mem_wd_fd < 0 || pcie_reg_wr_fd < 0 ) {
    fprintf(stderr, "error: failed to open PCIe device file: xdma0\n");
    exit(-1);
  }
  
  pgsz = sysconf(_SC_PAGESIZE);

}

uint32_t htif_xdma_t::read_write_reg_dev(addr_t target_addr, uint32_t val, bool wr_flag){
  off_t offset = target_addr & (pgsz - 1);
    off_t target_aligned = target_addr & (~(pgsz - 1)); 
  uint32_t read_result = 0;
  
  void *map = mmap(NULL, offset + 4, PROT_READ | PROT_WRITE, MAP_SHARED, pcie_reg_wr_fd,
                target_aligned);
  char *map_ptr = (char *)map;          
  if (map == (void *)-1) {
        printf("Memory 0x%lx mapped failed: %s.\n",
            target_addr, strerror(errno));
    
    // ~htif_xdma_t();
        exit(-1);
    }

  map_ptr += offset;
  if (wr_flag){
    val = htoll(val);
    *((uint32_t *) map_ptr) = val;
  }
  else{
    val = *((uint32_t *) map_ptr);
    val = ltohl(val);
  }

  map_ptr -= offset;
    if (munmap((void *)map_ptr, offset + 4) == -1) {
        printf("Memory 0x%lx mapped failed: %s.\n",
            target_addr, strerror(errno));
    }
  
  return val;
}

void htif_xdma_t::write_reg(addr_t target_addr, uint32_t writeval){
  read_write_reg_dev(target_addr, writeval, true);
}

uint32_t htif_xdma_t::read_reg(addr_t target_addr){
  return read_write_reg_dev(target_addr, 0, false);
}

htif_xdma_t::htif_xdma_t(const std::vector<std::string>& args) : htif_t(args) {
  open_dev();
  
  // request memory for read/write trunk;
  posix_memalign((void **)&read_allocated, MEMALIGN_SIZE /*alignment */ , 2 * MEMALIGN_SIZE);
  posix_memalign((void **)&write_allocated, MEMALIGN_SIZE /*alignment */ , 2 * MEMALIGN_SIZE);
  target = context_t::current();
  host.init(host_thread, this);
}


htif_xdma_t::~htif_xdma_t() {
  if (pcie_mem_rd_fd > 0)
    close(pcie_mem_rd_fd);
  if (pcie_mem_wd_fd > 0)
    close(pcie_mem_wd_fd);
  if (pcie_reg_wr_fd > 0)
    close(pcie_reg_wr_fd);
  if (pcie_mem_rd_fd1 > 0)
    close(pcie_mem_rd_fd1);
  
  if (read_allocated != NULL)
    free(read_allocated);
  if (write_allocated != NULL)
    free(write_allocated);
}

void htif_xdma_t::host_thread(void *arg)
{
  htif_xdma_t *pcie_t = static_cast<htif_xdma_t*>(arg);
  pcie_t->run();

  while (true)
    pcie_t->target->switch_to();
}

// Interrupt core 0 to make it start executing the program in DRAM
// write register val to hard that reset the hard mcu
void htif_xdma_t::reset()
{
    uint32_t a = 0x1234; //little endien
    // mcu stall wait write 0x1234 to 0x2011000 register mcu run
    write_reg(MCU_RESET_OVER_ADDR, a);
}

void htif_xdma_t::reset_board(){
    uint32_t one = 1;
    write_reg(CHIP_RESET_REG_ADDR, one);
}

ssize_t htif_xdma_t::read_to_buffer(const char *fname, int fd, char *buffer, uint64_t size,
            uint64_t base)
{
    ssize_t rc;
    uint64_t count = 0;
    char *buf = buffer;
    off_t offset = base;
    int loop = 0;

    while (count < size) {
        uint64_t bytes = size - count;

        if (bytes > RW_MAX_SIZE)
            bytes = RW_MAX_SIZE;

        if (offset) {
            rc = lseek(fd, offset, SEEK_SET);
            if (rc != offset) {
                fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
                    fname, rc, offset);
                perror("seek file");
                return -EIO;
            }
        }

        /* read data from file into memory buffer */
        rc = read(fd, buf, bytes);
        if (rc < 0) {
            fprintf(stderr, "%s, read 0x%lx @ 0x%lx failed %ld.\n",
                fname, bytes, offset, rc);
            perror("read file");
            return -EIO;
        }

        count += rc;
        if ((uint64_t)rc != bytes) {
            fprintf(stderr, "%s, read underflow 0x%lx/0x%lx @ 0x%lx.\n",
                fname, rc, bytes, offset);
            break;
        }

        buf += bytes;
        offset += bytes;
        loop++;
    }

    if (count != size && loop)
        fprintf(stderr, "%s, read underflow 0x%lx/0x%lx.\n",
            fname, count, size);
    return count;
}

ssize_t htif_xdma_t::write_from_buffer(const char *fname, int fd, char *buffer, size_t size,
            addr_t base)
{
    ssize_t rc;
    uint64_t count = 0;
    char *buf = buffer;
    off_t offset = base;
    int loop = 0;

    while (count < size) {
        uint64_t bytes = size - count;

        if (bytes > RW_MAX_SIZE)
            bytes = RW_MAX_SIZE;

        if (offset) {
            rc = lseek(fd, offset, SEEK_SET);
            if (rc != offset) {
                fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
                    fname, rc, offset);
                perror("seek file");
                return -EIO;
            }
        }

        /* write data to file from memory buffer */
        rc = write(fd, buf, bytes);
        if (rc < 0) {
            fprintf(stderr, "%s, write 0x%lx @ 0x%lx failed %ld.\n",
                fname, bytes, offset, rc);
            perror("write file");
            return -EIO;
        }

        count += rc;
        if ((uint64_t)rc != bytes) {
            fprintf(stderr, "%s, write underflow 0x%lx/0x%lx @ 0x%lx.\n",
                fname, rc, bytes, offset);
            break;
        }
        buf += bytes;
        offset += bytes;

        loop++;
    }   

    if (count != size && loop)
        fprintf(stderr, "%s, write underflow 0x%lx/0x%lx.\n",
            fname, count, size);

    return count;
}

// read data from hard use pcie
void htif_xdma_t::read_chunk(addr_t taddr, size_t nbytes, void* dst)
{
  read_to_buffer(XDMA_READ_DEVICE, pcie_mem_rd_fd, read_allocated, nbytes, taddr);
  memcpy(dst, read_allocated, nbytes);
  memset(read_allocated, 0, nbytes);
}

// write data to hard use pcie channel
void htif_xdma_t::write_chunk(addr_t taddr, size_t nbytes, const void* src)
{
  memcpy(write_allocated, src, nbytes);
  write_from_buffer(XDMA_WRIT_DEVICE, pcie_mem_wd_fd, write_allocated, nbytes, taddr);
  memset(write_allocated, 0, nbytes);
}

void htif_xdma_t::switch_to_host(void)
{
  host.switch_to();
}

void htif_xdma_t::switch_to_target(void)
{
  target->switch_to();
}

// check write data or read data is ready,not be used
void htif_xdma_t::tick(bool out_valid, uint32_t out_bits, bool in_ready)
{
  
}

void htif_xdma_t::read_board_ddr(addr_t taddr, size_t nbytes, void* dst)
{
    read_chunk(taddr, nbytes, dst);
}

void htif_xdma_t::write_board_ddr(addr_t taddr, size_t nbytes, const void* src){
    write_chunk(taddr, nbytes, src);
}

uint32_t htif_xdma_t::read_board_reg(addr_t target_addr){
    return read_reg(target_addr);
}
