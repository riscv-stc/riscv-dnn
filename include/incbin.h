#ifndef __INCBIN_H__
#define __INCBIN_H__

#define INCBIN(var, binfile, section) \
    extern uint8_t var[]; \
    __asm__(".section " section ", \"a\", @progbits\n\t" \
            ".global " #var "\n\t" \
            ".type " #var ", @object \n\t" \
            ".align 26 \n" \
            #var ":\n\t" \
            ".incbin  \"" binfile "\"\n\t")

        
#endif // __INCBIN_H__