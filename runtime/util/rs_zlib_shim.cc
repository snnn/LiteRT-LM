// Shim for prebuilt GemmaModelConstraintProvider on Linux.
//
// The prebuilt shared library references zlib symbols with an `rs_` prefix.
// Provide C wrappers that forward to the real zlib implementation.

#if defined(__linux__)

#include <zlib.h>

extern "C" {

uLong rs_adler32(uLong adler, const Bytef* buf, uInt len) {
  return adler32(adler, buf, len);
}

int rs_inflateInit2_(z_streamp strm, int windowBits, const char* version,
                     int stream_size) {
  return inflateInit2_(strm, windowBits, version, stream_size);
}

int rs_inflate(z_streamp strm, int flush) { return inflate(strm, flush); }

int rs_inflateEnd(z_streamp strm) { return inflateEnd(strm); }

}  // extern "C"

#endif  // defined(__linux__)
