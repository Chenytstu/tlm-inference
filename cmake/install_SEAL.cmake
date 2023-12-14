find_package(SEAL 3.3.2 EXACT QUIET PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
if (NOT SEAL_FOUND)
    message(STATUS "SEAL 3.3.2 was not found: clone and install SEAL locally")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/SEAL/native/src/CMakeLists.txt")
        find_package(Git REQUIRED)
        message(STATUS "initialize Git submodule: extern/SEAL")
        execute_process(COMMAND git submodule update --init --recursive extern/SEAL
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        execute_process(COMMAND git apply "${PROJECT_SOURCE_DIR}/cmake/seal.patch"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build .
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/native/src")
    execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/native/src")
    find_package(SEAL 3.3.2 EXACT REQUIRED PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
endif()


