//run with: pack_text_simple <in.txt> <out.bin>
//byte-level tokenizer where I:
// -remove UTF-8 BOM if there is (EF BB BF)
// -convert CRLF and CR to LF
// -write bytes to out.bin (each byte is a token id in 0-255)

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "run with: %s <in.txt> <out.bin>\n", argv[0]);
        return 2;
    }
    const char* in_path=argv[1];
    const char* out_path=argv[2];

    //read whole file into memory
    struct stat st;
    if (stat(in_path, &st) != 0) { 
	    perror("stat"); return 1; 
    }
    size_t n = (size_t)st.st_size;

    FILE* in = fopen(in_path, "rb");
    if (!in){ 
	    perror("fopen in"); return 1; 
    }

    uint8_t* buf = (uint8_t*)malloc(n ? n : 1);
    if (!buf){ 
	    fprintf(stderr, "OOM\n"); 
	    fclose(in); return 1; 
    }

    if (n && fread(buf, 1, n, in) != n){ 
	    perror("fread"); 
	    fclose(in); 
	    free(buf); 
	    return 1; 
    }
    fclose(in);

    //prepare output buffer (same size is enough; i only ever delete '\r')
    uint8_t* out = (uint8_t*)malloc(n ? n : 1);
    if (!out){ 
	    fprintf(stderr, "OOM\n"); 
	    free(buf); 
	    return 1; 
    }

    //strip UTF-8 BOM if there is
    size_t i=0;
    if (n >= 3 && buf[0]==0xEF && buf[1]==0xBB && buf[2]==0xBF){ 
	    i = 3;
    }

    //newline normalization & copy
    size_t o=0;
    while (i<n){
        uint8_t b = buf[i++];
        if (b == '\r'){
            //if it is CRLF, consume the '\n' too; either way output one '\n'
            if (i < n && buf[i] == '\n'){ 
		    i++;
	    }
            out[o++]='\n';
        }else{
            out[o++]=b;
        }
    }

    //write output
    FILE* fo = fopen(out_path, "wb");
    if (!fo){ 
	    perror("fopen out"); 
	    free(buf); 
	    free(out); 
	    return 1; 
    }
    if (o && fwrite(out, 1, o, fo) != o){ 
	    perror("fwrite"); 
	    fclose(fo); 
	    free(buf); 
	    free(out); 
	    return 1; 
    }
    fclose(fo);

    fprintf(stderr, "[pack] %s -> %s (%zu bytes -> %zu tokens)\n", in_path, out_path, n, o);
    free(buf);
    free(out);
    return 0;
}
