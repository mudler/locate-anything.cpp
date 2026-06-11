function(la_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE locate_anything)
  target_include_directories(${name} PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/third_party)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES
      SKIP_RETURN_CODE 77
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT "LA_TEST_GGUF=${CMAKE_SOURCE_DIR}/models/locate-anything-f32.gguf;LA_TEST_BASELINE=${CMAKE_SOURCE_DIR}/dumps/reference.gguf;LA_TEST_FIXTURES=${CMAKE_SOURCE_DIR}/dumps/vit_fixtures.gguf;LA_TEST_SLOW=${CMAKE_SOURCE_DIR}/dumps/reference_slow.gguf;LA_TEST_MTP=${CMAKE_SOURCE_DIR}/dumps/reference_mtp.gguf;LA_TEST_PREPROC=${CMAKE_SOURCE_DIR}/dumps/reference_preproc.gguf;LA_TEST_FAST=${CMAKE_SOURCE_DIR}/dumps/reference_fast.gguf")
endfunction()

la_add_test(test_smoke)
la_add_test(test_model_loader)
la_add_test(test_backend)
la_add_test(test_ggml_extend)
la_add_test(test_vit_posemb)
la_add_test(test_vit_encoder)
la_add_test(test_vit_rope)
la_add_test(test_projector)
la_add_test(test_lm)
la_add_test(test_decode)
la_add_test(test_mtp)
la_add_test(test_fast)
la_add_test(test_preproc)
la_add_test(test_tokenizer)
la_add_test(test_engine)
la_add_test(test_visualize)
la_add_test(test_capi)
la_add_test(test_quantize)

if(LA_BUILD_CLI AND TARGET locate-anything-cli)
  add_test(NAME test_cli COMMAND ${CMAKE_COMMAND} -E env
      LA_TEST_GGUF=${CMAKE_SOURCE_DIR}/models/locate-anything-f32.gguf
      LA_CLI_BIN=$<TARGET_FILE:locate-anything-cli>
      bash ${CMAKE_SOURCE_DIR}/tests/test_cli.sh)
  set_tests_properties(test_cli PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} SKIP_RETURN_CODE 77)
endif()
