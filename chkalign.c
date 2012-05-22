/* Compile with gcc -O chkalign.c -o chkalign */
/* Output on my machine is: bad alignment in blah */
/* Steven G. Johnson http://gcc.gnu.org/ml/gcc-bugs/1999-11/msg00167.html	*/

#include "Macros.h"
IDENTIFY( "Check alignment" );

struct yuck {
     int blechh;
};

int one(void)
{
     return 1;
}

struct yuck ick(void)
{
     struct yuck y;
     y.blechh = 3;
     return y;
}

void blah(int foo)
{
     double foobar;
     if ((((long) &foobar) & 0x7)) printf("bad alignment in blah\n");
}

int main(void)
{
     double okay1;
     struct yuck y;
     double okay2;
     if ((((long) &okay1) & 0x7)) printf("bad alignment in main\n");
     if ((((long) &okay2) & 0x7)) printf("bad alignment in main\n");
     y = ick();
     blah(one());
     return 0;
}


