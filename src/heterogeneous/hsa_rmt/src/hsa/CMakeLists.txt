include(../../CppLint.cmake)
set(CMAKE_SUPPRESS_REGENERATION TRUE)
cmake_minimum_required( VERSION 2.6.0 )

# Group samples by folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set( FOLDER_GROUP ${FOLDER_GROUP}/${FOLDER_NAME} )

set( FOLDER_NAME hsa )
set( SUBDIRECTORIES 
  fir_inter_beam
  fir_inter_errinj
  fir_intra_beam
  fir_intra_errinj
  page_rank_inter_beam
  page_rank_inter_errinj
  page_rank_intra_beam
  page_rank_intra_errinj
  kmeans_inter_beam
  kmeans_inter_errinj
  kmeans_intra_beam
  kmeans_intra_errinj
)

set( check_target ${FOLDER_NAME}_check )
set( check_target_src )
add_style_check_target( ${check_target} "${check_target_src}" 
  "${SUBDIRECTORIES}" )

###############################################################

foreach( subdir ${SUBDIRECTORIES} )
    add_subdirectory( ${subdir} )
endforeach( subdir )
