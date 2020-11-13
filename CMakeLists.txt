# Set some CMake properties    
CMAKE_MINIMUM_REQUIRED( VERSION 3.9 )


MESSAGE("====================")
MESSAGE("Configuring LBPM-slim")
MESSAGE("====================")


# Set the project name
SET( PROJ LBPM )          # Set the project name for CMake
SET( LBPM_LIB lbpm-slim )  # Set the final library name
SET( TEST_MAX_PROCS 16 )

# Initialize the project
PROJECT( ${PROJ} LANGUAGES CXX )

# Prevent users from building in place
IF ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}" )
    MESSAGE( FATAL_ERROR "Building code in place is a bad idea" )
ENDIF()

# Set the default C++ standard
SET( CMAKE_CXX_EXTENSIONS OFF )
IF ( NOT CMAKE_CXX_STANDARD )
    IF ( CXX_STD )
        MESSAGE( FATAL_ERROR "CXX_STD is obsolete, please set CMAKE_CXX_STANDARD" )
    ENDIF()
    SET( CMAKE_CXX_STANDARD 14 )
ENDIF()
IF ( ( "${CMAKE_CXX_STANDARD}" GREATER "90" ) OR ( "${CMAKE_CXX_STANDARD}" LESS "14" ) )
    MESSAGE( FATAL_ERROR "C++14 or newer required" )
ENDIF()

# Set source/install paths
SET( ${PROJ}_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
IF( INSTALL_DIR )
    SET( ${PROJ}_INSTALL_DIR "${${PROJ}_INSTALL_DIR}" )
ELSEIF( PREFIX )
    SET( ${PROJ}_INSTALL_DIR "${PREFIX}" )
ELSEIF( NOT ${PROJ}_INSTALL_DIR )
    SET( ${PROJ}_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}" )
ENDIF()
INCLUDE_DIRECTORIES( "${${PROJ}_INSTALL_DIR}/include" )
SET( CMAKE_MODULE_PATH ${${PROJ}_SOURCE_DIR} ${${PROJ}_SOURCE_DIR}/cmake )

# Include macros
INCLUDE( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros.cmake" )
INCLUDE( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libraries.cmake" )
INCLUDE( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LBPM-macros.cmake" )

# Set testing paramaters
ENABLE_TESTING()
INCLUDE( CTest )

CONFIGURE_SYSTEM()

# Add some directories to include
INCLUDE_DIRECTORIES( "${${PROJ}_INSTALL_DIR}/include" )

# Create custom targets for build-test, check, and distclean
ADD_CUSTOM_TARGET( build-test )
ADD_CUSTOM_TARGET( build-examples )
ADD_CUSTOM_TARGET( check COMMAND  make test  )
ADD_DISTCLEAN( analysis tests liblbpm-slim.* cpu gpu example common )


# Check for CUDA
CHECK_ENABLE_FLAG( USE_CUDA 0 )
NULL_USE( CMAKE_CUDA_FLAGS )
IF ( USE_CUDA )
    ADD_DEFINITIONS( -D USE_CUDA )
    ENABLE_LANGUAGE( CUDA )
ENDIF()


# Configure external packages

CONFIGURE_MPI()     # MPI must be before other libraries
CONFIGURE_NETCDF()
CONFIGURE_LBPM()
CONFIGURE_LINE_COVERAGE()
INCLUDE( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SharedPtr.cmake" )
CONFIGURE_SHARED_PTR( "${${PROJ}_INSTALL_DIR}/include" "std" )


# Macro to create 1,2,4 processor tests
MACRO( ADD_LBPM_TEST_1_2_4 EXENAME ${ARGN} )
    ADD_LBPM_TEST( ${EXENAME} ${ARGN} )
    ADD_LBPM_TEST_PARALLEL( ${EXENAME} 2 ${ARGN} )
    ADD_LBPM_TEST_PARALLEL( ${EXENAME} 4 ${ARGN} )
ENDMACRO()


# Add the src directories
BEGIN_PACKAGE_CONFIG( lbpm-slim-library )
ADD_PACKAGE_SUBDIRECTORY( common )
ADD_PACKAGE_SUBDIRECTORY( analysis )
ADD_PACKAGE_SUBDIRECTORY( IO )
#ADD_PACKAGE_SUBDIRECTORY( threadpool )
ADD_PACKAGE_SUBDIRECTORY( models )
IF ( USE_CUDA )
    ADD_PACKAGE_SUBDIRECTORY( gpu )
ELSE()
    ADD_PACKAGE_SUBDIRECTORY( cpu )
ENDIF()
INSTALL_LBPM_TARGET( lbpm-slim-library  )
ADD_SUBDIRECTORY( tests )
#ADD_SUBDIRECTORY( threadpool/test )
ADD_SUBDIRECTORY( example )
INSTALL_PROJ_LIB()


