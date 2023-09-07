#pragma once

#include <stdexcept> //std::runtime_error
#include <string>



namespace topaz {

#ifdef DEBUG
constexpr void runtime_assert(bool condition, const char* msg) {
    if (!condition) throw std::runtime_error(msg);
}
#else
constexpr CUDA_HOSTDEV void runtime_assert([[maybe_unused]] bool condition, [[maybe_unused]] const char* msg) {}
#endif

} // namespace Utils
