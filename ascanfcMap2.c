#include "config.h"

#if USE_AA_REGISTER == 4 || USE_AA_REGISTER == 5

#if USE_AA_REGISTER == 4
#	warning "using unsorted SymbolTable mechanism"
	IDENTIFY( "C dictionary lookups of valid pointers based on the venerable CXX SymbolTable" );
#elif USE_AA_REGISTER == 5
#	warning "using sorted SymbolTable mechanism"
	IDENTIFY( "C dictionary lookups of valid pointers based on a sorted version of the venerable CXX SymbolTable" );
#endif


/* 20050129: a minimal implementation of something like what the C++ <map> container does:
 \ a table lookup mechanism. This one is based on doubly linked circular lists of entries
 \ that associate an ascanf_Function pointer with its type. The search of the circular lists
 \ for a matching entry always starts at the last succesful result, OR at the last new entry,
 \ and changes direction on each hit. That is supposed to speed up searches by calling functions
 \ which always check the same few pointers.
 \ To speed up matters further, the whole table is distributed over an array of SYMBOLTABLE_LISTSIZE
 \ elements; each address <p> is added to bin <p> % SYMBOLTABLE_LISTSIZE .
 */

#include <stdio.h>
#include <stdlib.h>

#include "ascanf.h"

extern FILE *StdErr;

typedef struct SymbolTable{
	struct SymbolTable *car, *cdr;
	address32 ptr;
	ascanf_Function *af;
	ascanf_Function_type type;
} SymbolTable;

typedef struct SymbolTableList{
	SymbolTable *Table, *Current;
	int direction;
	unsigned long items;
#ifdef DEBUG
	unsigned long max;
#endif
} SymbolTableList;

#define SYMBOLTABLE_LISTSIZE	64
#define TableIndex(ptr)	((int)((uint32_t)(ptr) % SYMBOLTABLE_LISTSIZE))

SymbolTableList ascanf_AddressMap[SYMBOLTABLE_LISTSIZE];

static SymbolTable *find_ascanf_AddressEntry( address32 p )
{ SymbolTableList *listhead= &ascanf_AddressMap[TableIndex(p)];
  SymbolTable *entry= (listhead->Current)? listhead->Current : (listhead->Current= listhead->Table);
	if( entry ){
		if( entry->ptr!= p ){
#if USE_AA_REGISTER == 5
			if( p< entry->ptr ){
				listhead->direction= 0;
#else
			if( listhead->direction ){
#endif
				entry= entry->car;
				while( entry->ptr!= p && entry!= listhead->Current ){
					entry= entry->car;
				}
			}
			else{
#if USE_AA_REGISTER == 5
				listhead->direction= 1;
#endif
				entry= entry->cdr;
				while( entry->ptr!= p && entry!= listhead->Current ){
					entry= entry->cdr;
				}
			}
			if( entry->ptr== p ){
				listhead->Current= entry;
#if USE_AA_REGISTER == 4
				listhead->direction= !listhead->direction;
#endif
			}
			else{
				entry= NULL;
			}
		}
	}
	return(entry);
}

#if USE_AA_REGISTER == 5

static SymbolTable *find_ascanf_AddressEntry_ge( address32 p )
{ SymbolTableList *listhead= &ascanf_AddressMap[TableIndex(p)];
  SymbolTable *entry= (listhead->Current)? listhead->Current : (listhead->Current= listhead->Table);
	if( entry ){
		if( entry->ptr!= p ){
		  double diff= ((double) ((unsigned long)p))- ((double) ((unsigned long)entry->ptr)), mdiff= diff;
		  SymbolTable *best= entry;
			if( diff> 0 ){
				listhead->direction= 1;
				  /* cycle one, because now we're == listhead->Current  */
				entry= entry->cdr;
				  /* update the difference info: */
				diff= ((double) ((unsigned long)p))- ((double) ((unsigned long)entry->ptr));
				if( diff>= 0 && diff< mdiff ){
					mdiff= diff;
					best= entry;
				}
				  /* now loop. We could make this a do...while loop, in which case we'd be
				   \ able to include the above initial cycle inside the loop. But we'd have
				   \ to test for diff>0 twice, which is probably (a tad) more expensive.
				   */
				while( diff> 0 && entry!= listhead->Current ){
					entry= entry->cdr;
					diff= ((double) ((unsigned long)p))- ((double) ((unsigned long)entry->ptr));
					if( diff< mdiff ){
						mdiff= diff;
						best= entry;
					}
				}
			}
			else{
				listhead->direction= 0;
				  /* cycle one */
				entry= entry->car;
				  /* update diff. info */
				diff= ((double) ((unsigned long)p))- ((double) ((unsigned long)entry->ptr));
				if( diff<= 0 && diff> mdiff ){
					mdiff= diff;
					best= entry;
				}
				  /* now loop */
				while( diff< 0 && entry!= listhead->Current ){
					entry= entry->car;
					diff= ((double) ((unsigned long)p))- ((double) ((unsigned long)entry->ptr));
					if( diff> mdiff ){
						mdiff= diff;
						best= entry;
					}
				}
			}
			listhead->Current= entry= best;
		}
	}
	return(entry);
}
#endif

static ascanf_Function *find_ascanf_Address( address32 p )
{ SymbolTable *entry;
	if( (entry= find_ascanf_AddressEntry(p)) ){
		return(entry->af);
	}
	else{
		return(NULL);
	}
}

#ifdef DEBUG
static int called= 0;
static unsigned long registrations= 0, new_registrations= 0;

void print_AddressMap()
{ int i, n= 0;;
	fprintf( StdErr, "### ascanf_AddressMap[%d]:", SYMBOLTABLE_LISTSIZE );
	for( i= 0; i< SYMBOLTABLE_LISTSIZE; i++ ){
		if( ascanf_AddressMap[i].max ){
			n+= 1;
			fprintf( StdErr, " [%d=%lu/%lu]", i, ascanf_AddressMap[i].items, ascanf_AddressMap[i].max );
		}
	}
	fprintf( StdErr, " (%d (once) used)\n", n );
	fprintf( StdErr, "### %lu new of %lu total registrations\n", new_registrations, registrations );
}
#endif

#if USE_AA_REGISTER == 4

#	ifdef SAFE_FOR_64BIT
void register_ascanf_Address( ascanf_Function *the_af, address32 af )
#	else
void register_ascanf_Address( ascanf_Function *af )
#	endif
{ SymbolTableList *listhead= &ascanf_AddressMap[TableIndex(af)];
  SymbolTable *entry= find_ascanf_AddressEntry(af);
	if( af ){
#ifdef DEBUG
		registrations+= 1;
#endif
		if( !entry ){
		  SymbolTable *new= malloc( sizeof(SymbolTable) );
			if( new ){
#	ifdef SAFE_FOR_64BIT
				new->ptr= af;
				new->af= the_af;
				new->type= the_af->type;
#	else
				new->ptr= new->af= af;
				new->type= af->type;
#	endif
				if( listhead->Table ){
				  /* Doubly linked circular list: */
					new->car= listhead->Table->car;
					listhead->Table->car= new;
					new->cdr= listhead->Table;
					new->car->cdr= new;
				}
				else{
					new->car= new->cdr= new;
				}
				listhead->Table= new;
				listhead->Current= listhead->Table;
				listhead->items+= 1;
#ifdef DEBUG
				listhead->max= listhead->items;
				new_registrations+= 1;
				if( !called ){
					atexit( print_AddressMap );
					called= 1;
				}
#endif
			}
		}
		else if( entry->type!= af->type ){
			entry->type= af->type;
		}
#ifndef _AIX
		register_VariableName(af);
#endif
	}
}

#else

#	ifdef SAFE_FOR_64BIT
void register_ascanf_Address( ascanf_Function *the_af, address32 af )
#	else
void register_ascanf_Address( ascanf_Function *af )
#	endif
{ SymbolTableList *listhead= &ascanf_AddressMap[TableIndex(af)];
	if( af ){
	  SymbolTable *entry= find_ascanf_AddressEntry(af);
#ifdef DEBUG
		registrations+= 1;
#endif
		if( entry && entry->ptr== af ){
#	ifdef SAFE_FOR_64BIT
			entry->type= the_af->type;
#	else
			entry->type= af->type;
#	endif
		}
		else{
		  SymbolTable *new= malloc( sizeof(SymbolTable) );
			if( new ){
				  /* redo the lookup, now going for the closest existing element. This makes
				   \ adding a new pointer somewhat more expensive, but should reduce the cost of
				   \ modifying an existing entry somewhat.
				   */
				entry= find_ascanf_AddressEntry_ge(af);
#	ifdef SAFE_FOR_64BIT
				new->ptr= af;
				new->af= the_af;
				new->type= the_af->type;
#	else
				new->ptr= new->af= af;
				new->type= af->type;
#	endif
				if( entry ){
					if( (void*) entry->ptr > (void*) af ){
					  /* Doubly linked circular list: add "before" entry */
						new->car= entry->car;
						entry->car= new;
						new->cdr= entry;
						new->car->cdr= new;
					}
					else{
					  /* Doubly linked circular list: add "after" entry */
						new->car= entry;
						new->cdr= entry->cdr;
						entry->cdr= new;
						new->cdr->car= new;
					}
				}
				else{
					new->car= new->cdr= new;
				}
				if( !listhead->Table ){
					listhead->Table= new;
				}
				listhead->Current= new;
				listhead->items+= 1;
#ifdef DEBUG
				listhead->max= listhead->items;
				new_registrations+= 1;
				if( !called ){
					atexit( print_AddressMap );
					called= 1;
				}
#endif
			}
		}
#ifndef _AIX
		register_VariableName(af);
#endif
	}
}

#endif

ascanf_Function *verify_ascanf_Address( address32 p, ascanf_Function_type type )
{ SymbolTable *entry= find_ascanf_AddressEntry(p);
#ifdef DEBUG
	if( entry ){
		if( entry->type==type ){
			return(1);
		}
		else{
			fprintf( StdErr, "### verify_ascanf_Address(0x%lx): type 0x%lx is not requested type 0x%lx\n",
				p, entry->type, type
			);
		}
	}
	else{
		fprintf( StdErr, "### verify_ascanf_Address(0x%lx): unknown (unregistered) address!\n", p );
	}
	return(0);
#else
	if( entry && entry->type==type ){
		return( entry->af );
	}
	else{
		return( NULL );
	}
#endif
}


void delete_ascanf_Address( address32 af )
{ SymbolTable *entry= find_ascanf_AddressEntry(af);
	if( entry ){
	  SymbolTableList *listhead= &ascanf_AddressMap[TableIndex(af)];
		if( listhead->items== 1 ){
			listhead->Table= NULL;
		}
		else{
			entry->car->cdr= entry->cdr;
			entry->cdr->car= entry->car;
			listhead->Current= (listhead->direction)? entry->car : entry->cdr;
			if( listhead->Table== entry ){
				listhead->Table= listhead->Current;
			}
			listhead->items-= 1;
			memset( entry, 0, sizeof(SymbolTable) );
			free( entry );
		}
#ifndef _AIX
		delete_VariableName(af->name);
#endif
	}
}


#else

#ifdef __GNUC__
	__attribute__((used))
#endif
static char *unused="unused";

#endif
