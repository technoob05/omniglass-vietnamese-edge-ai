set(TARGET llama-mtmd-resident-server)
add_executable(${TARGET} resident_vlm_server.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common mtmd cpp-httplib Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib
)
