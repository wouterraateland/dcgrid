cmake_minimum_required(VERSION 3.17)

project(Weather LANGUAGES CUDA CXX)
set(EXEC_NAME Weather)

option(BUILD_WITH_OPENVDB "Build with OpenVDB" off)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

set(DCGRID_BIN_DIR "${CMAKE_CURRENT_LIST_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${DCGRID_BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${DCGRID_BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${DCGRID_BIN_DIR}")

file(GLOB_RECURSE SOURCES_DCGRID RELATIVE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "src/*.*"
)

add_executable(${EXEC_NAME} ${SOURCES_DCGRID})

target_include_directories(
	${EXEC_NAME} PUBLIC
	"${CMAKE_SOURCE_DIR}/src"
	"${PROJECT_SOURCE_DIR}/include/cuda"
	"${PROJECT_SOURCE_DIR}/include/imguiplot"
)

# CUDA
target_include_directories(
	${EXEC_NAME} PUBLIC
	"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/include"
)
list(
	APPEND EXTRA_LIBS
	"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib/x64/cudart.lib"
	"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib/x64/cuda.lib"
)

if(MSVC)
	# MP compilation
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP ")
	
	set_target_properties(
		${EXEC_NAME} PROPERTIES
		CUDA_RESOLVE_DEVICE_SYMBOLS on
		CUDA_SEPARABLE_COMPILATION on
	)
	
	# make bin directory
	add_custom_command(
		TARGET ${EXEC_NAME}
		PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory ${DCGRID_BIN_DIR}
	)
	
	# set VS environment dir
	set_target_properties(
		${EXEC_NAME} PROPERTIES
		VS_DEBUGGER_WORKING_DIRECTORY ${DCGRID_BIN_DIR}
	)
endif()

target_compile_options(${EXEC_NAME} PUBLIC
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELEASE>>:
		-Xcompiler /openmp
		-gencode=arch=compute_52,code=sm_52
		-gencode=arch=compute_61,code=sm_61
		-gencode=arch=compute_61,code=compute_61>
	$<$<COMPILE_LANGUAGE:CXX>:>
)

target_compile_options(${EXEC_NAME} PUBLIC
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELWITHDEBINFO>>:
		-Xcompiler /openmp
		-gencode=arch=compute_52,code=sm_52
		-gencode=arch=compute_61,code=sm_61
		-gencode=arch=compute_61,code=compute_61
	>
	$<$<COMPILE_LANGUAGE:CXX>:>
)
target_compile_options(${EXEC_NAME} PUBLIC
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:
		-G
		-Xcompiler /openmp
		-gencode=arch=compute_52,code=sm_52
		-gencode=arch=compute_61,code=sm_61
		-gencode=arch=compute_61,code=compute_61
	>
	$<$<COMPILE_LANGUAGE:CXX>:>
)

target_link_libraries(
	${EXEC_NAME}
	PRIVATE
		${EXTRA_LIBS}
)
