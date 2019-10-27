#.rst:
# AddDymods
# -----------
#
# (adapted from ECMAddTests
#
# Convenience functions for adding "dymods" (plugins).
#
# ::
#
#   add_dymods(<sources> LINK_LIBRARIES <library> [<library> [...]]
#                           [TARGET_NAMES_VAR <target_names_var>]
#                           [DYMOD_NAMES_VAR <dymod_names_var>])
#
# A convenience function for adding multiple plugins, each consisting of a
# single source file. For each file in <sources>, a plugin will be
# created (the name of which will be the basename of the source file). This
# will be linked against the libraries given with LINK_LIBRARIES.
#
# The TARGET_NAMES_VAR and DYMOD_NAMES_VAR arguments, if given, should specify a
# variable name to receive the list of generated target and test names,
# respectively. This makes it convenient to apply properties to them as a
# whole, for example, using set_target_properties() or  set_tests_properties().
#
# The generated target executables will have the effects of ecm_mark_as_test()
# (from the :module:`ECMMarkAsTest` module) applied to it.
#
# ::
#
#   add_dymod(<sources> LINK_LIBRARIES <library> [<library> [...]]
#                          [DYMOD_NAME <name>]
#                          [INCLUDE_DIRS <pattern>])
#
# This is a single-plygin form of add_dymods that allows multiple source files
# to be used for a single plugin. If using multiple source files, DYMOD_NAME must
# be given; this will be used for both the target and plugin names.
#
#
# Since pre-1.0.0.

#=============================================================================
# Copyright 2013 Alexander Richardson <arichardson.kde@gmail.com>
# Copyright 2015 Alex Merry <alex.merry@kde.org>
# Copyright 2019 R.J.V. Bertin <rjvbertin@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

include(CMakeParseArguments)

function(add_dymod)
  # TARGET_NAME_VAR and DYMOD_NAME_VAR are undocumented args used by
  # add_dymods
  set(oneValueArgs DYMOD_NAME TARGET_NAME_VAR DYMOD_NAME_VAR)
  set(multiValueArgs LINK_LIBRARIES INCLUDE_DIRS DEFINITIONS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(_sources ${ARG_UNPARSED_ARGUMENTS})
  list(LENGTH _sources _sourceCount)
  if(ARG_DYMOD_NAME)
    set(_targetname ${ARG_DYMOD_NAME})
  elseif(${_sourceCount} EQUAL "1")
    #use the source file name without extension as the plugin name
    get_filename_component(_targetname ${_sources} NAME_WE)
  else()
    #more than one source file passed, but no plugin name given -> error
    message(FATAL_ERROR "add_dymod() called with multiple source files but without setting \"DYMOD_NAME\"")
  endif()

  set(_pluginname ${_targetname})
  add_library(${_targetname} MODULE ${_sources})
  if (ARG_INCLUDE_DIRS)
      target_include_directories(${_targetname} PUBLIC ${ARG_INCLUDE_DIRS})
  endif()
  if (ARG_DEFINITIONS)
      target_compile_definitions(${_targetname} PUBLIC ${ARG_DEFINITIONS})
  endif()
  set_target_properties(${_targetname} PROPERTIES PREFIX "")
  set_target_properties(${_targetname} PROPERTIES SUFFIX ".dymod")
  target_link_libraries(${_targetname} ${ARG_LINK_LIBRARIES})
  if (ARG_TARGET_NAME_VAR)
    set(${ARG_TARGET_NAME_VAR} "${_targetname}" PARENT_SCOPE)
  endif()
  if (ARG_DYMOD_NAME_VAR)
    set(${ARG_DYMOD_NAME_VAR} "${_pluginname}" PARENT_SCOPE)
  endif()
endfunction()

function(add_dymods)
  set(oneValueArgs TARGET_NAMES_VAR DYMOD_NAMES_VAR)
  set(multiValueArgs LINK_LIBRARIES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(dymod_names)
  set(target_names)
  foreach(_dymod_source ${ARG_UNPARSED_ARGUMENTS})
    add_dymod(${_dymod_source}
      LINK_LIBRARIES ${ARG_LINK_LIBRARIES}
      TARGET_NAME_VAR target_name
      DYMOD_NAME_VAR dymod_name
    )
    list(APPEND _dymod_names "${dymod_name}")
    list(APPEND _target_names "${target_name}")
  endforeach()
  if (ARG_TARGET_NAMES_VAR)
    set(${ARG_TARGET_NAMES_VAR} "${_target_names}" PARENT_SCOPE)
  endif()
  if (ARG_DYMOD_NAMES_VAR)
    set(${ARG_DYMOD_NAMES_VAR} "${_dymod_names}" PARENT_SCOPE)
  endif()
endfunction()
