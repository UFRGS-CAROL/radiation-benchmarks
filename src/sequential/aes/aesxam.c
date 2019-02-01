#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <ctype.h>

#include <fcntl.h>
#include <unistd.h>

#include "aes.h"
#include<arpa/inet.h>
#include<sys/socket.h>
#include <stdlib.h>
#include <sys/timeb.h>

#ifdef LOGS
#include "log_helper.h"
#endif

/* A Pseudo Random Number Generator (PRNG) used for the     */
/* Initialisation Vector. The PRNG is George Marsaglia's    */
/* Multiply-With-Carry (MWC) PRNG that concatenates two     */
/* 16-bit MWC generators:                                   */
/*     x(n)=36969 * x(n-1) + carry mod 2^16                 */
/*     y(n)=18000 * y(n-1) + carry mod 2^16                 */
/* to produce a combined PRNG with a period of about 2^60.  */
/* The Pentium cycle counter is used to initialise it. This */
/* is crude but the IV does not need to be secret.          */

/* void cycles(unsigned long *rtn)     */
/* {                           // read the Pentium Time Stamp Counter */
/*     __asm */
/*     { */
/*     _emit   0x0f            // complete pending operations */
/*     _emit   0xa2 */
/*     _emit   0x0f            // read time stamp counter */
/*     _emit   0x31 */
/*     mov     ebx,rtn */
/*     mov     [ebx],eax */
/*     mov     [ebx+4],edx */
/*     _emit   0x0f            // complete pending operations */
/*     _emit   0xa2 */
/*     } */
/* } */

#define RAND(a,b) (((a = 36969 * (a & 65535) + (a >> 16)) << 16) + (b = 18000 * (b & 65535) + (b >> 16))  )

void fillrand(char *buf, int len) {
	static unsigned long a[2], mt = 1, count = 4;
	static char r[4];
	int i;

	if (mt) {
		mt = 0;
		/*cycles(a);*/
		a[0] = 0xeaf3;
		a[1] = 0x35fe;
	}

	for (i = 0; i < len; ++i) {
		if (count == 4) {
			*(unsigned long*) r = RAND(a[0], a[1]);
			count = 0;
		}

		buf[i] = r[count++];
	}
}

int encfile(FILE *fin, FILE *fout, aes *ctx, char* fn) {
	char inbuf[16], outbuf[16] = { 0x60, 0x53, 0x7f, 0x1c, 0xd4, 0x5c, 0x92,
			0x66, 0x26, 0x11, 0xc1, 0x8e, 0x5f, 0xd9, 0xb8, 0x69 };
	/*fpos_t*/unsigned int flen;
	unsigned long i = 0, l = 0;
	int k;
	//fillrand(outbuf, 16);           /* set an IV for CBC mode           */

	/*for(k=0;k<16;k++){
	 printf("0x%x,",(unsigned int)(outbuf[k] & 0xFF));
	 }*/
	fseek(fin, 0, SEEK_END); /* get the length of the file       */
	//fgetpos(fin, &flen);            /* and then reset to start          */
	flen = ftell(fin);
	fseek(fin, 0, SEEK_SET);
	fwrite(outbuf, 1, 16, fout); /* write the IV to the output       */
	//fillrand(inbuf, 1);             /* make top 4 bits of a byte random */
	//printf("0x%x,",(unsigned int)(inbuf[0] & 0xFF));
	inbuf[0] = 0x60;
	l = 15; /* and store the length of the last */
	/* block in the lower 4 bits        */
	inbuf[0] = ((char) flen & 15) | (inbuf[0] & ~15);

	while (!feof(fin)) /* loop to encrypt the input file   */
	{ /* input 1st 16 bytes to buf[1..16] */
		i = fread(inbuf + 16 - l, 1, l, fin); /*  on 1st round byte[0] */
		/* is the length code    */
		if (i < l)
			break; /* if end of the input file reached */

		for (i = 0; i < 16; ++i) /* xor in previous cipher text  */
			inbuf[i] ^= outbuf[i];

		encrypt(inbuf, outbuf, ctx); /* and do the encryption        */

		if (fwrite(outbuf, 1, 16, fout) != 16) {
			printf("Error writing to output file: %s\n", fn);
			return -7;
		}
		/* in all but first round read 16   */
		l = 16; /* bytes into the buffer            */
	}

	/* except for files of length less than two blocks we now have one  */
	/* byte from the previous block and 'i' bytes from the current one  */
	/* to encrypt and 15 - i empty buffer positions. For files of less  */
	/* than two blocks (0 or 1) we have i + 1 bytes and 14 - i empty    */
	/* buffer position to set to zero since the 'count' byte is extra   */

	if (l == 15) /* adjust for extra byte in the */
		++i; /* in the first block           */

	if (i) /* if bytes remain to be output */
	{
		while (i < 16) /* clear empty buffer positions */
			inbuf[i++] = 0;

		for (i = 0; i < 16; ++i) /* xor in previous cipher text  */
			inbuf[i] ^= outbuf[i];

		encrypt(inbuf, outbuf, ctx); /* encrypt and output it        */

		if (fwrite(outbuf, 1, 16, fout) != 16) {
			printf("Error writing to output file: %s\n", fn);
			return -8;
		}
	}

	return 0;
}

int decfile(FILE *fin, FILE *fout, aes *ctx, char* ifn, char* ofn) {
	char inbuf1[16], inbuf2[16], outbuf[16], *bp1, *bp2, *tp;
	int i, l, flen;

	if (fread(inbuf1, 1, 16, fin) != 16) /* read Initialisation Vector   */
	{
		printf("Error reading from input file: %s\n", ifn);
		return 9;
	}

	i = fread(inbuf2, 1, 16, fin); /* read 1st encrypted file block    */

	if (i && i != 16) {
		printf("\nThe input file is corrupt");
		return -10;
	}

	decrypt(inbuf2, outbuf, ctx); /* decrypt it                       */

	for (i = 0; i < 16; ++i) /* xor with previous input          */
		outbuf[i] ^= inbuf1[i];

	flen = outbuf[0] & 15; /* recover length of the last block and set */
	l = 15; /* the count of valid bytes in block to 15  */
	bp1 = inbuf1; /* set up pointers to two input buffers     */
	bp2 = inbuf2;

	while (1) {
		i = fread(bp1, 1, 16, fin); /* read next encrypted block    */
		/* to first input buffer        */
		if (i != 16) /* no more bytes in input - the decrypted   */
			break; /* partial final buffer needs to be output  */

		/* if a block has been read the previous block must have been   */
		/* full lnegth so we can now write it out                       */

		if (fwrite(outbuf + 16 - l, 1, l, fout) != (unsigned long) l) {
			printf("Error writing to output file: %s\n", ofn);
			return -11;
		}

		decrypt(bp1, outbuf, ctx); /* decrypt the new input block and  */

		for (i = 0; i < 16; ++i) /* xor it with previous input block */
			outbuf[i] ^= bp2[i];

		/* set byte count to 16 and swap buffer pointers                */

		l = i;
		tp = bp1, bp1 = bp2, bp2 = tp;
	}

	/* we have now output 16 * n + 15 bytes of the file with any left   */
	/* in outbuf waiting to be output. If x bytes remain to be written, */
	/* we know that (16 * n + x + 15) % 16 = flen, giving x = flen + 1  */
	/* But we must also remember that the first block is offset by one  */
	/* in the buffer - we use the fact that l = 15 rather than 16 here  */

	l = (l == 15 ? 1 : 0);
	flen += 1 - l;

	if (flen)
		if (fwrite(outbuf + l, 1, flen, fout) != (unsigned long) flen) {
			printf("Error writing to output file: %s\n", ofn);
			return -12;
		}

	return 0;
}

int s;
struct sockaddr_in server;
unsigned int buffer[4];
void setup_socket(char* ip_addr, int port) {
	s = socket(PF_INET, SOCK_DGRAM, 0);
	//memset(&server, 0, sizeof(struct sockaddr_in));
	//printf("port: %d",port);
	//printf("ip: %s", ip_addr);
	server.sin_family = AF_INET;
	server.sin_port = htons(port);
	server.sin_addr.s_addr = inet_addr(ip_addr);

}

void send_message(size_t size) {
	//printf("message sent\n");
	sendto(s, buffer, 4 * size, 0, (struct sockaddr *) &server, sizeof(server));
}

int main(int argc, char *argv[]) {
	//m5_checkpoint(0,0);
	FILE *fin = 0, *fout = 0;
	char *cp = argv[7], ch, key[32];
	int i = 0, by = 0, key_len = 0, err = 0;
	aes ctx[1];
	FILE *golden;
	int ex = 0;
	int conter = 0;
	//-----------------------------------------------------
	//LOG HELPER
	char *ip_addr = argv[1];
	unsigned int port = atoi(argv[2]);
	char *input_file_path = argv[3];
	char *output_file_path = argv[4];
	char *golden_file_path = argv[5];
	char mode = toupper(argv[6][0]);

	//-----------------------------------------------------

	//struct timeb start, end;
	//int diff;
	//printf("comecou\n");
	if (argc != 8 || (mode != 'D' && mode != 'E')) {
		printf(
				"usage: rijndael IP port in_filename out_filename [d/e] key_in_hex\n");
		exit(-1);
	}

#ifdef LOGS
	char test_info[1024];
	snprintf(test_info, 1024, "input_file:%s output_file:%s golden_file:%s mode:%c key_digit:%s", input_file_path, output_file_path, golden_file_path, mode, cp);
	start_log_file("SequentialAES", test_info);
#else
	setup_socket(ip_addr, port);
#endif

	if (!(fout = fopen(output_file_path, "wb"))) /* try to open the output file */
	{
		printf("The output file: %s could not be opened\n", output_file_path);
		exit(-6);
	}
	while (1) {
		//printf("lol\n");
		cp = argv[7]; /* this is a pointer to the hexadecimal key digits  */
		i = 0; /* this is a count for the input digits processed   */
		if ((golden = fopen(golden_file_path, "rb")) == NULL) {
			fprintf(stderr, "%s: can't open %s\n", argv[0], golden_file_path);
			exit(EXIT_FAILURE);
		}
		if (!(fin = fopen(input_file_path, "rb"))) /* try to open the input file */
		{
			printf("The input file: %s could not be opened\n", input_file_path);
			exit(-5);

		}
		while (i < 64 && *cp) /* the maximum key length is 32 bytes and   */
		{ /* hence at most 64 hexadecimal digits      */
			ch = toupper(*cp++); /* process a hexadecimal digit  */
			if (ch >= '0' && ch <= '9')
				by = (by << 4) + ch - '0';
			else if (ch >= 'A' && ch <= 'F')
				by = (by << 4) + ch - 'A' + 10;
			else /* error if not hexadecimal     */
			{
				printf("key must be in hexadecimal notation\n");
				exit(-2);
			}

			/* store a key byte for each pair of hexadecimal digits         */
			if (i++ & 1)
				key[i / 2 - 1] = by & 0xff;
		}
		//printf("ferf\n");
		if (*cp) {
			printf("The key value is too long\n");
			exit(-3);
		} else if (i < 32 || (i & 15)) {
			printf("The key length must be 32, 48 or 64 hexadecimal digits\n");
			exit(-4);
		}
		//printf("rer\n");
		key_len = i / 2;
		//ftime(&start);

		// ftime(&end);
		//diff = (int) (1000.0 * (end.time - start.time) + (end.millitm - start.millitm));
		// printf("\nOperation took %u milliseconds\n", diff);
		//printf("rrererer4\n");
#ifdef LOGS
		start_iteration();
#endif
		//printf("yuyuuuuuu\n");
		if (mode == 'E') { /* encryption in Cipher Block Chaining mode */
			set_key(key, key_len, enc, ctx);

			err = encfile(fin, fout, ctx, input_file_path);
		} else { /* decryption in Cipher Block Chaining mode */
			//printf("Start\n");
			set_key(key, key_len, dec, ctx);

			err = decfile(fin, fout, ctx, input_file_path, output_file_path);

		}
		//printf("3453245545\n");
		rewind(fout);
#ifdef LOGS
		end_iteration();
#endif
		int index = 0;
		char golden_buf;
		char out_buf;
		int SDC_flag = 0;
		size_t errors = 0;
		//printf("16");
		while (fread(&golden_buf, 1, 1, golden) == 1
				&& fread(&out_buf, 1, 1, fout) == 1) {
			if (golden_buf != out_buf) {
				//printf("ERROR!\n");
				if (SDC_flag == 0) {
					buffer[0] = 0xDD000000;
				} else {
					buffer[0] = 0xCC000000;
				}
				buffer[1] = (unsigned int) index;
				buffer[2] = (unsigned int) out_buf;
				SDC_flag = 1;
#ifdef LOGS
				char error_detail[200];
				snprintf(error_detail, 200, "INDEX:%d e:%c r:%c", i, golden_buf, out_buf);
				log_error_detail(error_detail);
				errors++;
#else
				send_message(3);
#endif
			}
			index++;
		}

		//printf("17");
		if (SDC_flag == 0) {
			//printf("OK!\n");
			buffer[0] = 0xAA000000;
#ifdef LOGS
			if(errors){
				log_error_count(errors);
			}
#else
			send_message(1);
#endif
		}
		fclose(golden);
		rewind(fout);
		fclose(fin);
		/*if(conter==100){
		 return 0;
		 }*/
		conter++;
	}

#ifdef LOGS
	end_log_file();
#endif
	return err;
}
