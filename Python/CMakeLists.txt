# FindPython components are case-sensitive!
find_package(Python COMPONENTS Interpreter Development NumPy)

if (Python_Interpreter_FOUND AND Python_NumPy_FOUND)
    configure_file(python_numpy.h.in ${CMAKE_CURRENT_BINARY_DIR}/python_numpy.h)

    add_dymod(Python.c AscanfCall.c DataSet.c ULabel.c
        DYMOD_NAME Python
        INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR} ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS}
        LINK_LIBRARIES ${Python_LIBRARIES} X11::X11)

endif(Python_Interpreter_FOUND)
