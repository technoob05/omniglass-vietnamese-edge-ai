# QCS8550 resident MiniCPM-V service target.
#
# The pinned llama.cpp-omni fork keeps the generic server sources but only
# declares its custom full-Omni server target.  This target restores the
# generic OpenAI-compatible multimodal server so a VLM model and its mmproj
# stay resident on the box between browser turns.
set(TARGET llama-server)
add_executable(${TARGET}
    main.cpp
    server.cpp
    server-http.cpp
    server-models.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/vendor
    ../mtmd
    ../omni
    ../omni/voxcpm2
)
target_link_libraries(${TARGET} PRIVATE
    server-context
    llama-common
    omni
    mtmd
    voxcpm2_runtime
    cpp-httplib
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
