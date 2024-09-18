
#include <cstring>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "htif_xdma_t.h"
#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <algorithm>
#include <fesvr/option_parser.h>
#include <fstream>
#define MCU_RESET_OVER_ADDR     0x2011000
const long unsigned int max_chunk = 0x10000;
htif_xdma_t *pcie_t = NULL;
typedef uint64_t reg_t;
static struct option const long_options[] = {
    {"help", no_argument, NULL, 'h'},
    {"init-dump", no_argument, NULL, 0},
    {"exit-dump", no_argument, NULL, 0},
    {"dump-path", no_argument, NULL, 0},
    {0, 0, 0, 0}
};

static void usage(const char *name)
{
    int i = 0;

    fprintf(stdout, "%s\n\n", name);
    fprintf(stdout, "usage: %s [OPTIONS]\n\n", name);

    fprintf(stdout, "  -%c (--%s) print usage help and exit\n",
        long_options[i].val, long_options[i].name);
    i++;

    fprintf(stdout, "  --init-dump=<m1,...>  Dump memory on init\n");
    i++;

    fprintf(stdout, "  --exit-dump=<m1,...>  Dump memory on exit\n");
    i++;

    fprintf(stdout, "  --dump-path           Path for files to dump memory [default .]\n");
    i++;
    exit(0);
}

void dump_mem(std::vector<std::string> &mems, std::string &path, bool space_end){
    char fname[512];
    char *read_allocated = NULL;
    posix_memalign((void **)&read_allocated, max_chunk /*alignment */ , 2 * max_chunk);
    
    for (const std::string& mem: mems){
        const std::regex re("(0x[0-9a-fA-F]+):(0x[0-9a-fA-F]+)");
        std::smatch match;
        if (!std::regex_match(mem, match, re)) {
            std::cout << "Invalid dump format " << mem << std::endl;
            exit(1);
        }
        auto start = std::stoul(match[1], nullptr, 16);
        auto len = std::stoul(match[2], nullptr, 16);

        snprintf(fname, sizeof(fname), "%s/%s@ddr.0x%lx_0x%lx.dat", path.c_str(), "output_mem",start, len);
        std::ofstream ofs(fname, std::ios::out);
        if (!ofs.is_open()) {
            std::cout << "Failed to open file." << std::endl;
            exit(1);
        }
        
        uint64_t of_len = 0;
        for (size_t pos = 0; pos < len; pos += max_chunk){
            of_len = std::min(max_chunk, len - pos);
            pcie_t->read_board_ddr(start + pos, of_len, (char*)read_allocated + pos);
            uint16_t data;
            char buf[5];
            for (addr_t offset = 0; offset < of_len; offset += 2) {
                data = *((uint16_t *)(read_allocated + offset));
                sprintf(buf, "%04x", data);
                ofs << buf;
                if ((offset + 2) % 128) {
                    ofs << " ";
                } else {
                    if (space_end) ofs << " ";
                    ofs << std::endl;
                }
            }
            memset(read_allocated, 0, of_len);
        }

        ofs.close();
    }

    if (read_allocated != NULL)
            free(read_allocated);
}

int pcie_tick(std::vector<std::string> &argv){

    if (!pcie_t)
    {
        pcie_t = new htif_xdma_t(argv);
        // reset chipyard
        pcie_t->reset_board();
        char val;
        while (1)
        {   
            // wait the board launch over
            val = pcie_t->read_board_reg(MCU_RESET_OVER_ADDR);
            if (val == -52)
                break;
        }
    }
    return 0;
}

static std::vector<std::string> make_strings(const char* arg)
{
  std::stringstream ss(arg);
  std::string item;
  std::vector<std::string> result;
  while (std::getline(ss, item, ',')) {
    result.push_back(std::move(item));
  }
  return result;
}

static void suggest_help()
{
  fprintf(stderr, "Try 'spike --help' for more information.\n");
  exit(1);
}

int main(int argc, char * argv[]){
    int opt;
    bool verbose_flag = false;

    std::vector<std::string> init_dump;
    std::vector<std::string> exit_dump;
    std::string dump_path = ".";
    option_parser_t parser;
    parser.help(&suggest_help);
    parser.option('h', "help", 0, [&](const char *s){usage(argv[0]);});
    parser.option(0, "init-dump", 1, [&](const char* s){init_dump = make_strings(s);});
    parser.option(0, "exit-dump", 1, [&](const char* s){exit_dump = make_strings(s);});
    parser.option(0, "dump-path", 1, [&](const char* s){dump_path = s;});

    auto argv1 = parser.parse(argv);
    std::vector<std::string> htif_args(argv1, (const char*const*)argv + argc);
    
    if (!*argv1)
        usage(argv[0]);

    pcie_tick(htif_args);
    pcie_t->switch_to_host();

    if (!init_dump.empty())
        dump_mem(init_dump, dump_path, true);

    if (!exit_dump.empty())
        dump_mem(exit_dump, dump_path, true);
    printf("run over.\n");
    return 0;
}
