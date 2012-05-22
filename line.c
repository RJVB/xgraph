#include <stdio.h>

main()
{ char buf[1024];
	if( fgets( buf, sizeof(buf), stdin ) ){
		fputs( buf, stdout );
		exit(0);
	}
	exit(1);
}
