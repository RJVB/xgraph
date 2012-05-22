#include "config.h"

// 20061104: we could use this as a general pointer-verification mechanism. Associate allocated size to the address.

#include <stdio.h>
#include <errno.h>

#include "xfree.h"

#ifdef USE_AA_REGISTER

#if USE_AA_REGISTER == 1

IDENTIFY( "C++ associative array (dictionary) lookups of valid pointers" );

#include <map>
// using namespace std;

#warning "Using standard map implementation"

#include "ascanf.h"

std::map<void *, ascanf_Function_type> ascanf_AddressMap;

void register_ascanf_Address( ascanf_Function *af )
{
	if( af ){
		ascanf_AddressMap[(void*)af] = af->type;
		register_VariableName(af);
	}
}

int verify_ascanf_Address( void *p, ascanf_Function_type type )
{
#ifdef DEBUG
  int n;
	if( (n=ascanf_AddressMap.count(p)) && ascanf_AddressMap[p]==type ){
		return(1);
	}
	else{
		if( n ){
			fprintf( StdErr, "### verify_ascanf_Address(0x%lx): type 0x%lx is not requested type 0x%lx\n",
				p, ascanf_AddressMap[p], type
			);
		}
		else{
			fprintf( StdErr, "### verify_ascanf_Address(0x%lx): unknown (unregistered) address!\n", p );
		}
		return(0);
	}
#else
	return( (ascanf_AddressMap.count(p) && ascanf_AddressMap[p]==type)? 1 : 0 );
#endif
}

void delete_ascanf_Address( ascanf_Function *af )
{
	if( af && ascanf_AddressMap.count(af) ){
		ascanf_AddressMap.erase(af);
		delete_VariableName(af->name);
#ifdef _DENSE_HASH_MAP_H_
		ascanf_AddressMap.resize(0);
#endif
	}
}


#elif USE_AA_REGISTER == 2

#if __GNUC_MINOR__ <= 99
	IDENTIFY( "C++ associative array (dictionary) lookups of valid pointers, google::dense_hash_map extension <unsigned long>" );
#	warning "Using google::dense_hash_map extension (void* cast to unsigned long)"
#	include <google/dense_hash_map>
#	define xghash_map	google::dense_hash_map
#else
	// <ext/hash_map> is obsolescent in gcc 4.3 and not standard anyway...
	IDENTIFY( "C++ associative array (dictionary) lookups of valid pointers, __gnu_cxx hash_map extension <unsigned long>" );
#	warning "Using __gnu_cxx hash_map extension (void* cast to unsigned long)"
#	include <ext/hash_map>
// 	using namespace __gnu_cxx;
#	define	xghash_map	__gnu_cxx::hash_map
#endif


#include "ascanf.h"

struct eqptr{  
	bool operator()(unsigned long s1, unsigned long s2) const
	{  
		return( s1==s2 );
	}
};

xghash_map<unsigned long, ascanf_Function_type> ascanf_AddressMap;

#ifdef _DENSE_HASH_MAP_H_
static int aAM_initialised= 0;

static void init_aAM()
{
	if( !aAM_initialised ){
		  // these two keys (variable names) should never occur:
		ascanf_AddressMap.set_empty_key(0);
		ascanf_AddressMap.set_deleted_key((unsigned long) -1);
		aAM_initialised = 1;
	}
}
#else
#	define init_aAM()	/**/
#endif

void register_ascanf_Address( ascanf_Function *af )
{
	if( af ){
		init_aAM();
		ascanf_AddressMap[(unsigned long)af] = af->type;
		register_VariableName(af);
	}
}


int verify_ascanf_Address( void *p, ascanf_Function_type type )
{
	return( (ascanf_AddressMap.count((unsigned long)p) && ascanf_AddressMap[(unsigned long)p]==type)? 1 : 0 );
}


void delete_ascanf_Address( ascanf_Function *af )
{
	init_aAM();
	if( af && ascanf_AddressMap.count((unsigned long)af) ){
		ascanf_AddressMap.erase((unsigned long)af);
		delete_VariableName(af->name);
#ifdef _DENSE_HASH_MAP_H_
		ascanf_AddressMap.resize(0);
#endif
	}
}

#elif USE_AA_REGISTER == 3

#if __GNUC_MINOR__ <= 99
	IDENTIFY( "C++ associative array (dictionary) lookups of valid pointers and other C++ stuff, google::dense_hash_map extension for void *" );
#	warning "Using google::dense_hash_map extension (void* \"native\" implementation)"
#	include <google/dense_hash_map>
#	include HASH_FUN_H
#	define	xghash_map	google::dense_hash_map
#	define	xghash		HASH_NAMESPACE::hash
#else
	// <ext/hash_map> is obsolescent in gcc 4.3 and not standard anyway...
	IDENTIFY( "C++ associative array (dictionary) lookups of valid pointers and other C++ stuff, __gnu_cxx hash_map extension for void *" );
#	warning "Using __gnu_cxx hash_map extension (void* \"native\" implementation)"
#	include <ext/hash_map>
// 	using namespace __gnu_cxx;
#	define	xghash_map	__gnu_cxx::hash_map
#	define	xghash		__gnu_cxx::hash
#endif


#include "ascanf.h"

struct eqptr{  
	bool operator()(address32 s1, address32 s2) const
	{  
		return( s1==s2 );
	}
};

#if 0
static long tested= 0;
#endif

#	ifndef SAFE_FOR_64BIT
	namespace __gnu_cxx
	{
		template<> struct hash< address32 >                                                       
		{                                                                                           
			size_t operator()( const address32 x ) const                                           
			{                                                                                         
				  /* This works for 32bit pointers: the address is its own hash */
#if 0
						if( ((ascanf_Function*)x)->name ){
						  char name[32];
							strncpy( name, ((ascanf_Function*)x)->name, 31 );
							name[31]= '\0';
							fprintf( StdErr, "%x = \"%s\"\n", x, name );
						}
						tested+= 1;
#endif
				return (size_t) x;                                              
			}                                                                                         
		};                                                                                          
	}
#endif


#ifdef SAFE_FOR_64BIT
	typedef struct ascanf_MapEntry{
		ascanf_Function_type type;
		ascanf_Function *af;
	} ascanf_MapEntry;
	xghash_map<address32, ascanf_MapEntry*, xghash<address32>, eqptr > ascanf_AddressMap3;

#else

xghash_map<address32, ascanf_Function_type, xghash<address32>, eqptr > ascanf_AddressMap;

#endif

#	ifndef SAFE_FOR_64BIT

#		ifdef _DENSE_HASH_MAP_H_
			static int aAM_initialised= 0;

			static void init_aAM()
			{
				if( !aAM_initialised ){
					  // these two keys (variable names) should never occur:
					ascanf_AddressMap.set_empty_key(0);
					ascanf_AddressMap.set_deleted_key((unsigned long) -1);
					aAM_initialised = 1;
				}
			}
#		else
#			define init_aAM()	/**/
#		endif

void register_ascanf_Address( ascanf_Function *af )
{
	if( af ){
		init_aAM();
		ascanf_AddressMap[af] = af->type;
		register_VariableName(af);
	}
}


ascanf_Function *verify_ascanf_Address( address32 p, ascanf_Function_type type )
{
#ifdef DEBUG
  int n;
	if( (n=ascanf_AddressMap.count(p)) && ascanf_AddressMap[p]==type ){
#if 0
		tested= 0;
#endif
		return( (ascanf_Function*) p );
	}
	else{
		if( n ){
			fprintf( StdErr, "### verify_ascanf_Address(0x%lx): type 0x%lx is not requested type 0x%lx\n",
				p, ascanf_AddressMap[p], type
			);
		}
		else{
			fprintf( StdErr, "### verify_ascanf_Address(0x%lx): unknown (unregistered) address!\n", p );
		}
		return(NULL);
	}
#else
	return( (ascanf_AddressMap.count(p) && ascanf_AddressMap[p]==type)?  (ascanf_Function*) p  : NULL );
#endif
}


void delete_ascanf_Address( address32 af )
{
	init_aAM();
	if( af && ascanf_AddressMap.count(af) ){
		ascanf_AddressMap.erase(af);
		delete_VariableName( ((ascanf_Function*)af)->name );
#ifdef _DENSE_HASH_MAP_H_
		ascanf_AddressMap.resize(0);
#endif
	}
}

#	else

#	include <string.h>

#	ifdef _DENSE_HASH_MAP_H_
	static int aAM_initialised= 0;

	static void init_aAM()
	{
		if( !aAM_initialised ){
			  // these two keys (variable names) should never occur:
			ascanf_AddressMap3.set_empty_key((address32)0);
			ascanf_AddressMap3.set_deleted_key((address32) -1);
			aAM_initialised = 1;
		}
	}
#	else
#		define init_aAM()	/**/
#	endif

	void register_ascanf_Address( ascanf_Function *af, address32 repr )
	{
		if( af ){
		  ascanf_MapEntry *entry;
		  
			init_aAM();
			entry = (ascanf_AddressMap3.count(repr))? ascanf_AddressMap3[repr]
						: (ascanf_MapEntry*) malloc(sizeof(ascanf_MapEntry));
			if( entry ){
				entry->type= af->type;
				entry->af= af;
				ascanf_AddressMap3[repr] = entry;
			}
			else{
				fprintf( StdErr, "register_ascanf_Address(\"%s\"): can't alloc a map entry (%s)\n", af->name, strerror(errno) );
			}
			register_VariableName(af);
		}
	}


	ascanf_Function *verify_ascanf_Address( address32 p, ascanf_Function_type type )
	{ ascanf_MapEntry *entry;
		return( (ascanf_AddressMap3.count(p) && (entry= ascanf_AddressMap3[p])->type==type)? entry->af : NULL );
	}


	void delete_ascanf_Address( address32 p )
	{
		init_aAM();
		if( p && ascanf_AddressMap3.count(p) ){
		  ascanf_MapEntry *entry= ascanf_AddressMap3[p];
			ascanf_AddressMap3.erase(p);
			delete_VariableName(entry->af->name);
#ifdef _DENSE_HASH_MAP_H_
			ascanf_AddressMap3.resize(0);
#endif
			free(entry);
		}
	}

#	endif


#elif USE_AA_REGISTER == 4
  /* not us */
#elif USE_AA_REGISTER == 5
  /* not us */

#else

#warning "Unknown USE_AA_REGISTER specifier!"
#endif

#endif

#include <list>

typedef std::list<ascanf_Function *> AFList;

static int _remove_LinkedArray_from_List( AFList **dst, ascanf_Function *af )
{ int r= 0;
	if( af && *dst ){
		(**dst).remove(af);
		if( (r= (**dst).size()) == 0 ){
			delete *dst;
			*dst= NULL;
		}
		return( r );
	}
	return(r);
}

int remove_LinkedArray_from_List( void **dst, ascanf_Function *af )
{
	return( _remove_LinkedArray_from_List( (AFList**) dst, af ) );
}

static ascanf_Function *_walk_LinkedArray_List( AFList **dst, AFList::iterator **p )
{ ascanf_Function *af= NULL;
  static AFList::iterator **last_p= NULL;
	if( *dst ){
		if( !*p ){
			*p= new AFList::iterator;
			**p= (**dst).begin();
		}
		if( *p ){
			if( **p!= (**dst).end() ){
				af= ***p;
				(**p)++;
				last_p= p;
			}
			else{
#if __GNUC__ >= 4 && __GNUC_MINOR__ > 0
				;
#else
				**p= NULL;
#endif
				delete *p;
				*p= NULL;
				last_p= NULL;
			}
		}
	}
	else if( p && p== last_p && *p ){
#if __GNUC__ >= 4 && __GNUC_MINOR__ > 0
		;
#else
		**p= NULL;
#endif
		delete *p;
		*p= NULL;
		last_p= NULL;
	}
	return(af);
}

ascanf_Function *walk_LinkedArray_List( void **dst, void **p )
{
	return( _walk_LinkedArray_List( (AFList **) dst, (AFList::iterator **) p ) );
}

static int _register_LinkedArray_in_List( AFList **dst, ascanf_Function *af )
{
	if( af ){
		if( !*dst ){
			*dst= new AFList;
		}
		if( *dst ){
		  ascanf_Function *aaf;
		  AFList::iterator *iter=NULL;
		  int listed= 0;
			while( !listed && (aaf= _walk_LinkedArray_List(dst, &iter)) ){
				if( aaf == af ){
					listed+= 1;
				}
			}
			if( iter ){
				delete iter;
			}
			if( !listed ){
				(**dst).push_front(af);
			}
			return( (**dst).size() );
		}
	}
	return(0);
}

int register_LinkedArray_in_List( void **dst, ascanf_Function *af )
{
	return( _register_LinkedArray_in_List( (AFList**) dst, af ) );
}


#include <map>
// using namespace std;

#if __GNUC_MINOR__ <= 99
#	include <google/dense_hash_map>
#	include <google/dense_hash_map>
#	include HASH_FUN_H
#	define	xghash_map2	google::dense_hash_map
#	define	xghash2		HASH_NAMESPACE::hash
#else
// <ext/hash_map> is obsolescent in gcc 4.3 and not standard anyway...
#	include <ext/hash_map>
#	define	xghash_map2	__gnu_cxx::hash_map
#	define	xghash2		__gnu_cxx::hash
#endif
#include <string>
// using namespace __gnu_cxx;

// a method for telling if 2 entries are equal:
struct string_eqptr{  
	bool operator()(char* s1, char* s2) const
	{  
		return( strcmp(s1, s2)==0 );
	}
};

#include "dymod.h"

// 20100622: moving to google::dense_hash_map exposed a subtle bug: the name key stored in the hash_map is a pointer to
// the string passed in. If this string gets deallocated, or if this string lives in a shared library that got unloaded,
// accessing the key may lead to a bad access (sigsegv or sigbus) ... and this may happen even while doing a lookup of
// an unrelated name. Therefore, we now store a VNRentry, which contains a *copy* of the name string, and it is this copy
// that is used as the key. The VNRentry can be deallocated cleanly in delete_VariableName().
// 20101021 TOBEFINISHED: keep track of symbols from a DyMod (in case the library gets unloaded without removing all its
// entries properly).
typedef struct VNRentries{
	char *name;
	ascanf_Function *af;
#ifdef XG_DYMOD_SUPPORT
	struct DyModLists *dymod;
#endif
} VNRentries;

/* std::map<std::string, VNRentries* > Name2VariableHTable;	*/
/* xghash_map2<std::string, VNRentries* > Name2VariableHTable;	*/
xghash_map2<char *, VNRentries*, xghash2<char*>, string_eqptr > Name2VariableHTable;

// 20100610: "use" should be read as "maintain". Even with use_VariableNamesRegistry unset, lookups
// are attempted, so registered variables are found quickly.
static int use_VariableNamesRegistry= 0;

#ifdef _DENSE_HASH_MAP_H_
static int VNR_initialised= 0;

static void init_VNR()
{
	if( !VNR_initialised ){
	  static char *empty_key = (char*) "[]", *deleted_key = (char*) "][";
		  // these two keys (variable names) should never occur:
		Name2VariableHTable.set_empty_key( empty_key );
		Name2VariableHTable.set_deleted_key( deleted_key );
		VNR_initialised = 1;
	}
}
#else
#	define init_VNR()	/**/
#endif

int register_VariableNames( int yesno )
{ int ret= use_VariableNamesRegistry;
	use_VariableNamesRegistry= yesno;
	return(ret);
}

void register_VariableName( ascanf_Function *af )
{
	if( use_VariableNamesRegistry && af && af->name ){
	  VNRentries *entry = (VNRentries*) calloc( 1, sizeof(VNRentries) );
		if( entry && (entry->name = strdup(af->name)) ){
			init_VNR();
			entry->af = af;
#ifdef XG_DYMOD_SUPPORT
			entry->dymod = af->dymod;
#endif
			  /* 20070629: prevent double registrations... */
			delete_VariableName(entry->name);
			Name2VariableHTable[entry->name] = entry;
		}
	}
}


ascanf_Function *get_VariableWithName( char *name, int exhaustive )
{ ascanf_Function *af = NULL;
	if( name ){
		  // always attempt a lookup in the registry
		init_VNR();
		if( Name2VariableHTable.count(name) ){
		  VNRentries *entry = Name2VariableHTable[name];
			if( entry ){
				if( entry->dymod && entry->dymod->type == DM_Unloaded ){
					fprintf( StdErr, "get_VariableWithName(\"%s\"): variable left dangling from unloaded module!\n",
						name
					);
					goto bail;
				}
				af = entry->af;
				if( af ){
					if( !af->name ){
						fprintf( StdErr, "get_VariableWithName(\"%s\"): pruning entry with NULL name from lookup table!\n",
							name
						);
						Name2VariableHTable.erase(name);
						xfree(entry);
					}
					else if( !(af->name[0] && name[0]) || strcmp(af->name, name) ){
					  /* 20090922: Some sanity checks and cleanup */
						delete_VariableName(name);
						af = NULL;
					}
				}
			}
		}
#ifdef XGRAPH
		else if( !use_VariableNamesRegistry || exhaustive ){
		  // 20100610: if we use the registry, limit us to registered variables. If we don't use it,
		  // attempt a brute-force lookup with the function also used by the parser/compiler.
		  double dum;
			af = find_ascanf_function( name, &dum, NULL, (char*) "get_VariableWithName()" );
		}
#endif
	}
bail:
	return( af );
}


void delete_VariableName( char *name )
{ int n= -1;
  static unsigned short N=0;
#ifdef DEBUG
  unsigned short nn = 0, in = 0;
#endif
	if( name && use_VariableNamesRegistry ){
		init_VNR();
		while( (n= Name2VariableHTable.count(name))> 0 ){
		  VNRentries *entry = Name2VariableHTable[name];
			Name2VariableHTable.erase(name);
			xfree(entry->name);
			xfree(entry);
			N += 1;
#ifdef DEBUG
			if( !in ){
				in = n;
			}
			nn += 1;
#endif
		}
#ifdef DEBUG
		if( nn || in ){
			fprintf( StdErr, "## Removed %hu (%hu) instances of \"%s\"\n", nn, in, name );
		}
#endif
#ifdef _DENSE_HASH_MAP_H_
		if( N >= 32 ){
			Name2VariableHTable.resize(0);
			N = 0;
		}
#endif
	}
}




std::map<double, long> Double2IndexTable;

// a method for telling if 2 entries are equal:
struct d2l_eqptr{  
	bool operator()(double s1, double s2) const
	{  
#ifdef _DENSE_HASH_MAP_H_
		  // 20100616: since we use 2 different NaN values as the empty and deleted values, we need to do a
		  // byte-compare, not a value compare.
		return( !memcmp( (void*) &s1, (void*) &s2, sizeof(double) ) );
#else
		return( s1==s2 );
#endif
	}
};

#if (__GNUC__ > 3) && (__GNUC_MINOR__ < 3)
// a method for hashing doubles. Do very simple: cast to a float (which takes care of losing some precision)
// and then interpret that float as a size_t. We'll have to hope that sizeof(size_t)==sizeof(float) (== 4).
// Alternatively, put an unsigned long xxl in the union, and cast that to the size_t return value.
// For making a histogram ("density spectrum") of a large array of random values, this hashed version is at least
// twice faster than the standard map (on a 1.5Ghz G4).
namespace __gnu_cxx
{
	template<> struct hash< double >                                                       
	{                                                                                           
		uint32_t operator()( const double x ) const                                           
		{                                                                                         
			union{
				float xx;
				uint32_t xxl;
			} hval;

			hval.xx= (float) x;
			return hval.xxl;                                              
#ifdef DEBUG
			fprintf( StdErr, "hash(%s) -> %g -> 0x%lx\n", ad2str(x, d3str_format, NULL), hval.xx, hval.xxl );
#endif
		}                                                                                         
	};                                                                                          
}
#else
	// gcc 4.3 and higher use tr1/functional for the hash functions, and those already have an appropriate method.
#endif

xghash_map2<double, long, xghash2<double>, d2l_eqptr > Double2IndexHTable;

#ifdef _DENSE_HASH_MAP_H_

#include "NaN.h"
static int DWI_initialised= 0;

static void init_DWI()
{
	if( !DWI_initialised ){
	  double nan;
#ifdef DEBUG
	  extern int PrintNaNCode;
#endif
		set_NaN(nan);
		  // these two keys should never occur:
		Double2IndexHTable.set_empty_key(nan);
#ifdef DEBUG
		PrintNaNCode=1;
		fprintf( StdErr, "init_DWI(): set_empty_key(%s)\n", ad2str(nan, d3str_format, NULL) );
#endif
		  // make a -NaN:
		I3Ed(nan)->s.s = 1;
		Double2IndexHTable.set_deleted_key(nan);
#ifdef DEBUG
		fprintf( StdErr, "init_DWI(): set_deleted_key(%s)\n", ad2str(nan, d3str_format, NULL) );
#endif
		DWI_initialised = 1;
	}
}
#else
#	define init_DWI()	/**/
#endif

void register_DoubleWithIndex( double value, long idx )
{
	init_DWI();
	Double2IndexHTable[value] = idx;
}


long get_IndexForDouble( double value )
{
	return( (Double2IndexHTable.count(value))? Double2IndexHTable[value] : -1 );
}


void delete_IndexForDouble( double value )
{ static unsigned short N=0;
	init_DWI();
	if( Double2IndexHTable.count(value) ){
		Double2IndexHTable.erase(value);
		N += 1;
	}
#ifdef _DENSE_HASH_MAP_H_
	if( N >= 32 ){
		Double2IndexHTable.resize(0);
	}
#endif
}

// 20120414: file descriptor -> file pointer hashmapping
// a method for telling if 2 entries are equal:
struct int_eqptr{  
	bool operator()(int s1, int s2) const
	{  
		return( s1==s2 );
	}
};

xghash_map2<int, FILE*, xghash2<int>, int_eqptr > fd2fpHTable;

#ifdef _DENSE_HASH_MAP_H_

static int D2P_initialised= 0;

static void init_D2P()
{
	if( !D2P_initialised ){
		// these two keys should never occur as valid file descriptors:
		fd2fpHTable.set_empty_key(-2);
		fd2fpHTable.set_deleted_key(-1);
		D2P_initialised = 1;
	}
}
#else
#	define init_D2P()	/**/
#endif

FILE *register_FILEsDescriptor( FILE *fp )
{
	if( fp ){
		init_D2P();
		fd2fpHTable[fileno(fp)] = fp;
	}
	return fp;
}


FILE *get_FILEForDescriptor( int fd )
{
	init_D2P();
	return( (fd2fpHTable.count(fd))? fd2fpHTable[fd] : NULL );
}


void delete_FILEsDescriptor( FILE *fp )
{ static unsigned short N=0;
	if( fp ){
	  int fd = fileno(fp);
		init_D2P();
		if( fd2fpHTable.count(fd) ){
			fd2fpHTable.erase(fd);
			N += 1;
		}
#ifdef _DENSE_HASH_MAP_H_
		if( N >= 32 ){
			fd2fpHTable.resize(0);
		}
#endif
}
}

