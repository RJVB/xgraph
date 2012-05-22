
#ifndef Py_OBJECTS_H
#define Py_OBJECTS_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct PyAscanfObject {
	PyObject_HEAD
	ascanf_Function *af;
	struct PAO_Options *opts;
} PyAscanfObject;

#ifndef __CYGWIN__
	PyAPI_DATA(PyTypeObject) PyAscanfObject_Type;
#else
// 20100331: On cygwin (possibly mingw32 too?), the PyAPI_DATA adds __declspec(dllimport) in front of the
// 'standard' extern declaration. That may be needed in some situations, but not in ours. Curiously,
// this attribute does not cause issues for function declarations...
	extern PyTypeObject PyAscanfObject_Type;
#endif

#ifdef IS_PY3K
#	define PyAscanfObject_Check(op) (Py_TYPE(op) == &PyAscanfObject_Type)
#else
#	define PyAscanfObject_Check(op) ((op)->ob_type == &PyAscanfObject_Type)
#endif

PyAPI_FUNC(PyObject *) PyAscanfObject_FromAscanfFunction( ascanf_Function *af );


/* Retrieve a pointer to an ascanf_Function object from a PyAscanfObject. */
PyAPI_FUNC(ascanf_Function *) PyAscanfObject_AsAscanfFunction(PyObject *);

/* Modify an ascanf_Function object. Fails (==0) if object has a destructor. */
PyAPI_FUNC(int) PyAscanfObject_SetAscanfFunction(PyObject *self, ascanf_Function *af );

typedef struct PyDataSetObject {
	PyObject_HEAD
	struct DataSet *set;
	struct PDO_Options *opts;
} PyDataSetObject;

#ifndef __CYGWIN__
	PyAPI_DATA(PyTypeObject) PyDataSetObject_Type;
#else
// 20100331: see comment above.
	extern PyTypeObject PyDataSetObject_Type;
#endif

#ifdef IS_PY3K
#	define PyDataSetObject_Check(op) (Py_TYPE(op) == &PyDataSetObject_Type)
#else
#	define PyDataSetObject_Check(op) ((op)->ob_type == &PyDataSetObject_Type)
#endif


PyAPI_FUNC(PyObject *) PyDataSetObject_FromDataSet( struct DataSet *set );

PyAPI_FUNC(struct DataSet *) PyDataSetObject_AsDataSet(PyObject *);

PyAPI_FUNC(int) PyDataSetObject_SetDataSet(PyObject *self, struct DataSet *set );


#ifdef __cplusplus
}
#endif
#endif /* !Py_OBJECTS_H */
