if (USE_NNAPI AND ANDROID)
  set(Caffe2_CONTRIB_NNAPI_CPU_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/dlnnapi.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/nnapi_model_loader.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/nnapi_op.cc"
  )

  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS}
    ${Caffe2_CONTRIB_NNAPI_CPU_SRC} PARENT_SCOPE)
endif()
