#ifndef hip_check_h
#define hip_check_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// HIP headers
#include <hip/hip_runtime.h>

inline void hip_check(const char* file, int line, const char* cmd, hipError_t result) {
    if (__builtin_expect(result == hipSuccess, true))
        return;

    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "HIP_CHECK(" << cmd << ");\n";
    out << hipGetErrorName(result) << ": " << hipGetErrorString(result);
    throw std::runtime_error(out.str());
}

#define HIP_CHECK(ARG) (hip_check(__FILE__, __LINE__, #ARG, (ARG)))

#endif  // hip_check_h
