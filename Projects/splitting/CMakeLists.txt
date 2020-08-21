#Generate the local Makefile
if(CREATE_LOCAL_MAKEFILES)
  file(RELATIVE_PATH ZIRAN_RELATIVE_PATH ${ZIRAN_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  configure_file(${ZIRAN_SOURCE_DIR}/Scripts/Makefile.in ${CMAKE_CURRENT_SOURCE_DIR}/Makefile @ONLY)
endif()

if(OPENVDB_FOUND)

  set(CMAKE_CXX_FLAGS " -fPIC -Wall -Wextra -Wno-unused-parameter -Wcast-align -Wformat=2 -Winit-self -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -march=native -fno-math-errno")
  include_directories(${CMAKE_SOURCE_DIR}/Projects/amgclmpm)

  add_ziran_executable(splitting main.cpp SplittingSimulationWeakFormFast.cpp SplittingSimulationSolidSystemFast.cpp SplittingBoundaryParticles.cpp)
  target_compile_options(splitting PRIVATE ${ZIRAN_CXXFLAGS_LTO})
  target_link_libraries(splitting ziran mpmdev tbbmalloc tbbmalloc_proxy ${ZIRAN_CXXFLAGS_LTO})
  
endif()

