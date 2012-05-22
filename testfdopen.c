#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* This program will write a random integer to a file and read it back via the same stream */

int main(int argc, char** argv)
{
    
    int myInt = -1;

      srand(time(NULL));
    int ran = rand();

      char tnam[64]= "/tmp/XG-SimpleEdt-XXXXXX";
      int fd= mkstemp(tnam);
    FILE *pFile = fdopen(fd, "w+");

    char editbuf[256];
    
  //setvbuf ( pFile , NULL , _IONBF , 0 );
    
    fprintf(pFile, "%d", ran);
    fflush(pFile);
    fprintf(stderr, "Wrote to File\n\n");
    fprintf(stderr, "EOF = %d\n", feof(pFile));
    fprintf(stderr, "ERROR = %d\n\n", ferror(pFile));

    sprintf( editbuf, "xterm -e vi %s", tnam );
    system( editbuf );
    /* This actually reminds of of A/UX and maybe a few other old unixes, where modifications made to a running
       sh script also were not taken into account. But the mechanism may be completely different, though.
     */

    rewind(pFile);
    fprintf(stderr, "After Rewind\n\n");
    fprintf(stderr, "EOF   = %d\n", feof(pFile));
    fprintf(stderr, "ERROR = %d\n", ferror(pFile));
    fprintf(stderr, "Pos   = %d\n\n", ftell(pFile));
//     fscanf(pFile, "%d", &myInt);
    fgets( editbuf, sizeof(editbuf)-1, pFile );
    fprintf(stderr, "We should read: %d\n\n", ran);
    fprintf(stderr, "We read       : %s\n", editbuf);

  fclose (pFile);
  unlink(tnam);

  return 0;


}

