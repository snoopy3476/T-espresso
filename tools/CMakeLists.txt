add_executable(cutracedump cutracedump.c ../lib/TraceIO.h ../lib/Common.h)

install(FILES ${LLVM_BINARY_DIR}/bin/cutracedump
  DESTINATION bin
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
  GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)
