#include <stdio.h>
#include <stdlib.h>

main()
{ FILE *fp;
	fprintf( stderr, "< Opening pipe; fp=0x%lx\n", (fp= popen("sh -x", "w")) );
	if( fp ){
	  int d;
		fprintf( stderr, "< Writing command 1 to pipe\n" );
		fprintf( fp, "date\n" );
		fprintf( stderr, "< Writing command 2 to pipe\n" );
		fprintf( fp, "ps -f\n" );
		fprintf( stderr, "< Closing pipe\n" );
		d= pclose(fp);
		fprintf( stderr, "< Returned: %d\n", d ); 
	}
}
