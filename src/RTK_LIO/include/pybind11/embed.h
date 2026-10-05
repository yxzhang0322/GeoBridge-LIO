#pragma once

#include "pybind11.h"
#include <mutex>

namespace pybind11 {

class scoped_interpreter {
public:
    scoped_interpreter() {
        static std::once_flag flag;
        std::call_once(flag, []() {
            if (!Py_IsInitialized()) {
                Py_Initialize();
#if PY_VERSION_HEX < 0x030d0000
                PyEval_InitThreads();
#endif
            }
        });
    }
    ~scoped_interpreter() = default;
};

}
