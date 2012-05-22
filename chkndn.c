#include <stdio.h>

typedef union{
	char *t;
	long l;
} ua;

typedef union{
	char t[5];
	long l;
} ub;

#ifdef i386
int EndianType=1;
#else
int EndianType=0;
#endif

CheckEndianness()
{ union{
	char t[2];
	short s;
  } e;
	e.t[0]= 'a';
	e.t[1]= 'b';
	if( e.s== 0x6162 ){
		EndianType= 0;
	}
	else if( e.s== 0x6261 ){
		EndianType= 1;
	}
	else{
		fprintf( stderr, "Found weird endian type: *((short*)\"ab\")!= 0x6162 nor 0x6261 but 0x%hx\n",
			e.s
		);
		EndianType= 2;
	}
}

#ifdef CHECK_CONSTANT_ASSGN

	/* Activate this block to check whether your compiler/system allows
	 \ assignment of a string to a char[n] "static string" field of a global
	 \ variable. pgcc 2.91.66 and 2.95.0 under linux mandrake 6.1 don't allow
	 \ this when -fwritable-strings. Sort of understandable that they complain
	 \ about initialisation with a non-constant... It is allowed, however, to
	 \ do this when the variable is local instead of global (see the kkk2 var in
	 \ main() ).
	 */
typedef struct kk{
	double foo;
	char text[16];
} kk;

kk kkk= { 1.2, { 'a', 'l', 'l', 'o', 'w', 'e', 'd', '?'} };
kk kkk2= { 1.2, "allowed?" };
#endif

main()
{ ua a;
  ub b;
  char *msg[]= { "normal: your CPU looks in the same direction as you do :)", "reversed (lemmy guess.. Intel x86?)", "weird, go see your hardware vendor!" };
  char *t= "wine";
  long v= *((long*)t);
  int n= 0;
#ifdef CHECK_CONSTANT_ASSGN
  kk kkk2= { 1.2, "allowed?" };
#endif
  double ddd[]= {1.2, 3.4, 4.5, 0, 6.7, 0};
  char *sss[2];

	a.l= 0x61626162;
	b.t[4]= '\0';
	b.l= 0x554e2a58;
	n+= printf( "Do you like %s (%c%c%c%c)?\n", b.t, &v );
	switch( v ){
		case 'wine':
			n+= printf( "And how about wine?\n" );
			break;
		case 'eniw':
			n+= printf( "And how about eniw? Too much, I see - seeing backwards!\n" );
			break;
		default:
			n+= printf( "You have strange tastes about wine...\n" );
			break;
	}
	CheckEndianness();
	n+= printf( "Your byte order seems to be %s.\n", msg[EndianType] );
	{ struct kkk { unsigned long first, second; } *d= (struct kkk*) &ddd[3];
		d->first= (unsigned long) t;
		d->second= (unsigned long) msg[EndianType];
	}
	n+= printf( "0x%lx=t=%s\n", t,t);
	n+= vprintf( "Here is an array of 3 doubles + 2 strings printed with vprintf(): {%g,%g,%g}, %s, %s\n", ddd );
	sss[0]= t;
	sss[1]= msg[EndianType];
	n+= vprintf( "Here are the same 2 strings printed with vprintf(): %s, %s\n", sss );
	n+= printf( "And all this wasted an enormous amount of output bytes: %-4d...\n", n+64 );
	exit(n);
}
