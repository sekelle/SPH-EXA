
if(CMAKE_CUDA_COMPILER)
    add_library(planet_gpu OBJECT planet_gpu.cu accretion_gpu.cu)
    target_include_directories(planet_gpu PRIVATE ${CSTONE_DIR} ${SPH_DIR})
    target_compile_definitions(planet_gpu PRIVATE USE_CUDA)
    target_link_libraries(planet_gpu PUBLIC CUDA::cudart)
endif()
