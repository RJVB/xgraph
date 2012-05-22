#ifndef _PY_INITMODULE_H

#ifdef IS_PY3K

PyObject *Py_InitModule( const char *mName, PyMethodDef *methodDef )
{ static struct PyModuleDef moduleDef;
	// 20120412: PyModuleDef_HEAD_INIT just sets the 1st member bits to all 0
	memset( &moduleDef, 0, sizeof(moduleDef) );
	moduleDef.m_name = mName;
	moduleDef.m_size = -1;
	moduleDef.m_methods = methodDef;
	return PyModule_Create(&moduleDef);
}

PyObject *Py3_InitModule( struct PyModuleDef *moduleDef, const char *mName, PyMethodDef *methodDef )
{
	if( !moduleDef ){
		return NULL;
	}
	memset( moduleDef, 0, sizeof(*moduleDef) );
	moduleDef->m_name = mName;
	moduleDef->m_size = -1;
	moduleDef->m_methods = methodDef;

	return PyModule_Create(moduleDef);
}

PyObject *Py_InitModule3( const char *mName, PyMethodDef *methodDef, const char *mDoc )
{ static struct PyModuleDef moduleDef;
	// 20120412: PyModuleDef_HEAD_INIT just sets the 1st member bits to all 0
	memset( &moduleDef, 0, sizeof(moduleDef) );
	moduleDef.m_name = mName;
	moduleDef.m_doc = mDoc;
	moduleDef.m_size = -1;
	moduleDef.m_methods = methodDef;
	return PyModule_Create(&moduleDef);
}

PyObject *Py3_InitModule3( struct PyModuleDef *moduleDef, const char *mName, PyMethodDef *methodDef, const char *mDoc )
{
	if( !moduleDef ){
		return NULL;
	}
	memset( moduleDef, 0, sizeof(*moduleDef) );
	moduleDef->m_name = mName;
	moduleDef->m_doc = mDoc;
	moduleDef->m_size = -1;
	moduleDef->m_methods = methodDef;

	return PyModule_Create(moduleDef);
}

#endif

#define _PY_INITMODULE_H
#endif
