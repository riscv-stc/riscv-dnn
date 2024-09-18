#ifndef __HTIF_PCIE_T_H
#define __HTIF_PCIE_T_H

#include "htif.h"
#include "context.h"

#include <string>
#include <vector>
#include <deque>
#include <stdint.h>
#include <fcntl.h>
#define SAI_CMD_READ 0
#define SAI_CMD_WRITE 1

#define SAI_ADDR_CHUNKS 2
#define SAI_LEN_CHUNKS 2

class htif_xdma_t : public htif_t
{
 public:
  htif_xdma_t(const std::vector<std::string>& args);
  virtual ~htif_xdma_t();

  void switch_to_host();
  
  void tick(bool out_valid, uint32_t out_bits, bool in_ready);
  void read_board_ddr(addr_t taddr, size_t nbytes, void* dst);
  void write_board_ddr(addr_t taddr, size_t nbytes, const void* src);
  void reset_board();
  uint32_t read_board_reg(addr_t target_addr);
 protected:
  void reset() override;
  void read_chunk(addr_t taddr, size_t nbytes, void* dst) override;
  void write_chunk(addr_t taddr, size_t nbytes, const void* src) override;
  void switch_to_target();

  size_t chunk_align() override { return 8; }
  size_t chunk_max_size() override { return 4096; }

 private:
  context_t host;
  context_t* target;
  off_t pgsz;  // page size
  char *write_allocated = NULL;
  char *read_allocated = NULL;
  static void host_thread(void *tsi);
  void open_dev();
  int pcie_mem_rd_fd;
  int pcie_mem_wd_fd;
  int pcie_reg_wr_fd;
  int pcie_mem_rd_fd1;

  ssize_t read_to_buffer(const char *fname, int fd, char *buffer, uint64_t size,
            uint64_t base);
  ssize_t write_from_buffer(const char *fname, int fd, char *buffer, size_t size,
            addr_t base);
  uint32_t read_write_reg_dev(addr_t target_addr, uint32_t val, bool wr_flag);
  void write_reg(addr_t target_addr, uint32_t writeval);
  uint32_t read_reg(addr_t target_addr);
};

#endif

