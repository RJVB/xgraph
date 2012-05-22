#ifndef _PYTHONINTERFACE_H

typedef struct DM_Python_Interface{
	DyModTypes type;

	int (*isAvailable)();

	  /* Run_Python_Expr calls PyRun_SimpleString(expr) and returns its return-code */
	int (*Run_Python_Expr)( char *expr );
	  /* imports <filename> through PyRun_AnyFile(), after setting the arguments list argv[0] to the filename.
	   \ <filename> can be a pipe command to be read from.
	   \ When unlink_afterwards is True, the file is unlinked (removed) after reading from it.
	   \ <sourceFile> can be set to the name of the actual file that the code comes from (in case it transits
	   \ via a temporary file, one that might get unlinked after reading it, for instance).
	   \ 20120413: if a Python3 interpreter is loaded, passing ispy2k=True will cause the filename to be converted
	   \ from Python2 syntax.
	   */
	int (*Import_Python_File)( char *filename, char *sourceFile, int unlink_afterwards, int ispy2k );

	  /* Get_Python_ReturnValue retrieves the value of the ascanf.ReturnValue Python variable
	   \ and returns 1 upon success. Return code 0 signals an error, in which case <value_return>
	   \ will not have been touched.
	   */
	int (*Get_Python_ReturnValue)( double *value_return );

	  /* Evaluate_Python_Expr evaluates expr and returns its value or, if that's not possible,
	   \ ascanf.ReturnValue. It returns 1 upon success and 0 upon error, in which case
	   \ <result_return> will not have been touched.
	   */
	int (*Evaluate_Python_Expr)( char *expr, double *result_return );

	  /* This does same thing as the ascanf function Python-Shell[]; the return value from Run_Python_Expr()
	   \ is in the optional <result> argument if open_PythonShell returns True. Call open_PythonShell(NULL,...)
	   \ to mimick the argument-less ascanf call Python-Shell .
	   */
	int (*open_PythonShell)( double *arg, int *result );

	int (*ascanf_PythonCall)( ascanf_Function *pfunc, int argc, double *args, double *result_return );

	void (*Python_SysArgv)( int argc, char *argv[] );

	void (*Python_INCREF)( void *pobj );
	void (*Python_DECREF)( void *pobj );

	int (*Python_CheckSignals)();
	void (*Python_SetInterrupt)();

	int *pythonActive;

} DM_Python_Interface;


#define _PYTHONINTERFACE_H
#endif

