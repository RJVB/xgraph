/* Copyright (C) 1991, 1992, 1996, 1997, 1998 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If not,
   write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

#ifdef	BSD
#include </usr/include/stdio.h>
#else
#include <stdio.h>
#include <stdarg.h>
#endif
#include <stdlib.h>
#include <string.h>


int tvfscanf( FILE *s, char *format, ... )
{ va_list ap;
	va_start(ap, format);
	return( vfscanf( s, format, ap) );
}

int tvsscanf( char *s, char *format, ... )
{ va_list ap;
	va_start(ap, format);
	return( vsscanf( s, format, ap) );
}

int main (int argc, char **argv)
{
  char buf[BUFSIZ];
  FILE *in = stdin, *out = stdout;
  int x;
  int result = 0;

  if (sscanf ("0", "%d", &x) != 1)
    {
      fputs ("test failed!\n", stdout);
      result = 1;
    }

  sscanf ("conversion] Zero flag Ze]ro#\n", "%*[^]] %[^#]\n", buf);
  if (strcmp (buf, "] Zero flag Ze]ro") != 0)
    {
      fputs ("test failed!\n", stdout);
      result = 1;
    }

  if (argc == 2 && !strcmp (argv[1], "-opipe"))
    {
      out = popen ("/bin/cat", "w");
      if (out == NULL)
	{
	  perror ("popen: /bin/cat");
	  result = 1;
	}
    }
  else if (argc == 3 && !strcmp (argv[1], "-ipipe"))
    {
      sprintf (buf, "/bin/cat %s", argv[2]);
      in = popen (buf, "r");
      if (in == NULL)
	{
	  perror ("popen: /bin/cat");
	  result = 1;
	}
    }

  {
    char name[50];
    fprintf (out,
	     "sscanf (\"thompson\", \"%%s\", name) == %d, name == \"%s\"\n",
	     sscanf ("thompson", "%s", name),
	     name);
  }

  fputs ("Testing fscanf(in,\"%d%f%lf%s\", &i, &x, &d, name, 0)\n", out);

  fputs ("Test 1a: fscanf()\n", out);
  {
    int n, i;
    float x;
    double d;
    char name[50];
    n = fscanf (in, "%d%f%lf%s", &i, &x, &d, name, 0);
    fprintf (out, "n = %d, i = %d, x = %f, d= %g, name = \"%.50s\"\n", n, i, x, d, name);
    if (n != 3 ){
	fputs ("read failed (n!=3)!\n", stdout);
	result = 1;
      }
  }
  fprintf (out, "Residual: \"%s\"\n", fgets (buf, sizeof (buf), in));
  if (strcmp (buf, "\n"))
    {
      fputs ("test failed!\n", stdout);
      result = 1;
    }

  fputs ("Test 1b: vfscanf()\n", out);
  {
    int n, i;
    float x;
    double dd,d;
    char name[50];
    n = tvfscanf (in, "%lf%d%f%lf%s", &dd, &i, &x, &d, name, 0);
    fprintf (out, "n = %d, dd= %g, i = %d, x = %f, d= %g, name = \"%.50s\"\n", n, dd, i, x, d, name);
    if (n != 3 ){
	fputs ("read failed (n!=3)!\n", stdout);
	result = 1;
      }
  }
  fprintf (out, "Residual: \"%s\"\n", fgets (buf, sizeof (buf), in));
  if (strcmp (buf, "\n"))
    {
      fputs ("test failed!\n", stdout);
      result = 1;
    }

  fputs ("Test 2: vfscanf( in, \"%2d%f%*d %[0123456789]\", &i, &x, name, 0)\n", out );
  {
    int i;
    float x;
    char name[50];
    tvfscanf( in, "%2d%f%*d %[0123456789]", &i, &x, name, 0);
    fprintf (out, "i = %d, x = %f, name = \"%.50s\"\n", i, x, name);
  }
  fprintf (out, "Residual: \"%s\"\n", fgets (buf, sizeof (buf), in));
  if (strcmp (buf, "a72\n"))
    {
      fputs ("test failed!\n", stdout);
      result = 1;
    }

  fputs ("Test 3: sscanf()\n", out);
  {
    int res, val, n;

    res = sscanf ("-242", "%3o%n", &val, &n);
    printf ("res = %d, val = %d, n = %d\n", res, val, n);
    if (res != 1 || val != -20 || n != 3)
      {
	fputs ("test failed!\n", stdout);
	result = 1;
      }
  }

  fputs ("Test 4: vsscanf()\n", out);
  {
    int res, val, n;

    res = tvsscanf ("-242", "%3o%n", &val, &n, 0);
    printf ("res = %d, val = %d, n = %d\n", res, val, n);
    if (res != 1 || val != -20 || n != 3)
      {
	fputs ("test failed!\n", stdout);
	result = 1;
      }
  }

  fputs ("Test 5a: sscanf\n", out);
  {
    double a = 0, b = 0;
    int res, n;

    res = sscanf ("1234567", "%3lg%3lg%n", &a, &b, &n);
    printf ("res = %d, a = %g, b = %g, n = %d\n", res, a, b, n);

    if (res != 2 || a != 123 || b != 456 || n != 6)
      {
	fputs ("test failed!\n", stdout);
	result = 1;
      }

    res = sscanf ("0", "%lg", &a);
    printf ("res = %d, a = %g\n", res, a);

    if (res != 1 || a != 0)
      {
	fputs ("test failed!\n", stdout);
	result = 1;
      }

    res = sscanf ("1e3", "%lg%n", &a, &n);
    printf ("res = %d, a = %g, n = %d\n", res, a, n);

    if (res != 1 || a != 1000 || n != 3)
      {
	fputs ("test failed!\n", stdout);
	result = 1;
      }
  }

  fputs ("Test 5b: vsscanf\n", out);
  {
    double a = 0, b = 0;
    int res, n;

    res = tvsscanf ("1234567", "%3lg%3lg%n", &a, &b, &n, 0);
    printf ("res = %d, a = %g, b = %g, n = %d\n", res, a, b, n);

    if (res != 2 || a != 123 || b != 456 || n != 6)
      {
	fputs ("test failed!\n", stdout);
	result = 1;
      }

    res = sscanf ("0", "%lg", &a);
    printf ("res = %d, a = %g\n", res, a);

    if (res != 1 || a != 0)
      {
	fputs ("test failed!\n", stdout);
	result = 1;
      }

    res = sscanf ("1e3", "%lg%n", &a, &n);
    printf ("res = %d, a = %g, n = %d\n", res, a, n);

    if (res != 1 || a != 1000 || n != 3)
      {
	fputs ("test failed!\n", stdout);
	result = 1;
      }
  }

  fputs ("Test 6:\n", stdout);
  {
    char *p = (char *) -1;
    int res;

    sprintf (buf, "%p", NULL);
    res = sscanf (buf, "%p", &p);
    printf ("sscanf (\"%s\", \"%%p\", &p) = %d, p == %p\n", buf, res, p);

    if (res != 1 || p != NULL)
      {
	fputs ("test failed!\n", stdout);
	result = 1;
      }
  }

  exit (result);
}
