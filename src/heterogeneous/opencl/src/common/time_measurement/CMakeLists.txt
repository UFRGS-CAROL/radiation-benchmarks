set( FOLDER_NAME time_measurement )
set( SUBDIRECTORIES )
set( SRC_FILES
  time_keeper_impl.cc
  time_keeper_summary_printer.cc
  time_measurement_impl.cc
  timer_impl.cc)
set( HEADER_FILES 
  time_keeper.h
  time_keeper_impl.h
  time_keeper_summary_printer.h
  time_measurement.h
  time_measurement_impl.h
  timer.h
  timer_impl.h)
set( TEST_FILES
  time_keeper_impl_test.cc)

# Style check
set( check_target ${FOLDER_NAME}_check )
set( check_target_src ${SRC_FILES} ${HEADER_FILES} ${TEST_FILES})
add_style_check_target(${check_target} 
  "${check_target_src}" "${SUBDIRECTORIES}")

# Unit test
if (enable-test)
  add_executable(${FOLDER_NAME}_test 
    ${TEST_FILES} ${SRC_FILES} ${HEADER_FILES})
  target_link_libraries(${FOLDER_NAME}_test gtest gtest_main)
  add_test(${FOLDER_NAME} ${FOLDER_NAME}_test)
endif (enable-test)

###############################################################

set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )
project( ${FOLDER_NAME} )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

add_library(${FOLDER_NAME} ${SRC_FILES} ${HEADER_FILES})
