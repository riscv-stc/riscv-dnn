
#ifndef __RESNET50V2_WEIGHTDATA_NEW_
#define __RESNET50V2_WEIGHTDATA_NEW_

#include <stdint.h>
#include "../../../include/incbin.h"

int32_t woffset[] = [0, 9408, 29888, 33984, 70848, 91328, 107712, 144576, 165056, 181440, 218304, 238784, 386240, 435392, 656576, 730304, 828608, 1049792, 1123520, 1221824, 1443008, 1516736, 1615040, 1836224, 1909952, 2467008, 2630848, 3368128, 3646656, 3974336, 4711616, 4990144, 5317824, 6055104, 6333632, 6661312, 7398592, 7677120, 8004800, 8742080, 9020608, 9348288, 10085568, 10364096, 12526784, 13116608, 15770816, 16852160, 18031808, 20686016, 21767360, 22947008, 25601216, 26682560, 28732608];
int32_t aoffset[] = [0, 128, 256, 384, 896, 1024, 1152, 1664, 1792, 1920, 2432, 2688, 2944, 3968, 4224, 4480, 5504, 5760, 6016, 7040, 7296, 7552, 8576, 9088, 9600, 11648, 12160, 12672, 14720, 15232, 15744, 17792, 18304, 18816, 20864, 21376, 21888, 23936, 24448, 24960, 27008, 28032, 29056, 33152, 34176, 35200, 39296, 40320, 41344];
int32_t boffset[] = [64, 192, 320, 640, 960, 1088, 1408, 1728, 1856, 2176, 2560, 2816, 3456, 4096, 4352, 4992, 5632, 5888, 6528, 7168, 7424, 8064, 8832, 9344, 10624, 11904, 12416, 13696, 14976, 15488, 16768, 18048, 18560, 19840, 21120, 21632, 22912, 24192, 24704, 25984, 27520, 28544, 31104, 33664, 34688, 37248, 39808, 40832, 43392];
extern uint8_t weight_data[];
INCBIN(weight_data, "weight.bin",  ".scdata.params");
extern uint8_t alpha_data[];
INCBIN(alpha_data, "alpha.bin",  ".scdata.params");

#endif