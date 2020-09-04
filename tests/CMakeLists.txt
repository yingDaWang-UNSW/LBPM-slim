# Copy files for the tests

ADD_LBPM_EXECUTABLE( lbpm_color_simulator )
ADD_LBPM_EXECUTABLE( lbpm_permeability_simulator )
ADD_LBPM_EXECUTABLE( lbpm_dfh_simulator )
ADD_LBPM_EXECUTABLE( lbpm_serial_decomp )
ADD_LBPM_EXECUTABLE( lbpm_morphopen_pp )
#ADD_LBPM_EXECUTABLE( lbpm_uCT_pp )
# Add the tests
ADD_LBPM_TEST( TestFluxBC )
ADD_LBPM_TEST( TestMap )
#ADD_LBPM_TEST( TestMRT )
#ADD_LBPM_TEST( TestColorGrad )
ADD_LBPM_TEST( TestColorGradDFH )
#ADD_LBPM_TEST( TestBubbleDFH ../example/Bubble/input.db)
#ADD_LBPM_TEST( TestColorMassBounceback ../example/Bubble/input.db)
ADD_LBPM_TEST( TestPressVel  ../example/Bubble/input.db)
ADD_LBPM_TEST( TestPoiseuille ../example/Piston/poiseuille.db)
ADD_LBPM_TEST( TestForceMoments  ../example/Bubble/input.db)
ADD_LBPM_TEST( TestForceD3Q19 )
ADD_LBPM_TEST( TestMomentsD3Q19 )
#ADD_LBPM_TEST( TestInterfaceSpeed  ../example/Bubble/input.db)
ADD_LBPM_TEST( TestMassConservationD3Q7 ../example/Bubble/input.db)
ADD_LBPM_TEST_PARALLEL( TestSegDist 8 )
ADD_LBPM_TEST_PARALLEL( TestCommD3Q19 8 )
ADD_LBPM_TEST_1_2_4( testCommunication )
ADD_LBPM_TEST_1_2_4( testUtilities )

# Sample test that will run with 1, 2, and 4 processors, failing with 4 or more procs
ADD_LBPM_TEST_1_2_4( hello_world )
#ADD_LBPM_TEST( TestColorBubble ../example/Bubble/input.db)
#ADD_LBPM_TEST( TestColorSquareTube ../example/Bubble/input.db)

ADD_LBPM_TEST( TestColorBubble ../example/Bubble/input.db)
ADD_LBPM_TEST( TestColorSquareTube ../example/Bubble/input.db)

SET_TESTS_PROPERTIES( hello_world        PROPERTIES ENVIRONMENT "MPICH_RDMA_ENABLED_CUDA=0")
IF ( USE_MPI ) 
    SET_TESTS_PROPERTIES( hello_world_2procs PROPERTIES ENVIRONMENT "MPICH_RDMA_ENABLED_CUDA=0")
    SET_TESTS_PROPERTIES( hello_world_4procs PROPERTIES ENVIRONMENT "MPICH_RDMA_ENABLED_CUDA=0")
ENDIF()

# Add CPU/GPU specific test
IF ( USE_CUDA )
    ADD_SUBDIRECTORY( gpu )
ELSE()
    ADD_SUBDIRECTORY( cpu )
ENDIF()
