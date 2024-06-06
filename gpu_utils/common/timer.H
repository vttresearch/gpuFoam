#pragma once

#include "cuda_host_dev.H"
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stddef.h>
#include <sys/time.h>


static inline CUDA_HOSTDEV double second() {
#ifdef __NVIDIA_BACKEND__


    return static_cast<double>(clock());

#else

    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<double>(tv.tv_sec) +
           static_cast<double>(tv.tv_usec) * 1.0e-6;

#endif
}

static inline CUDA_HOSTDEV char* my_strcpy(char*       dest,
                                           const char* src) {
    int i = 0;
    do { dest[i] = src[i]; } while (src[i++] != 0);
    return dest;
}

static inline CUDA_HOSTDEV int my_strcmp(const char* s1,
                                         const char* s2) {
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    return *reinterpret_cast<const unsigned char*>(s1) -
           *reinterpret_cast<const unsigned char*>(s2);
}
struct Timer {

    using char_array = std::array<char, 10>;

    static constexpr size_t MAX_INSTANCES = 30;

    CUDA_HOSTDEV
    Timer()
        : m_label_count(0)
        , m_labels()
        , m_starts()
        , m_times() {}

    CUDA_HOSTDEV void start(const char* label) {

        int idx = find_index(label);

        // New label
        if (idx == -1) {

            my_strcpy(&(m_labels[m_label_count][0]), label);
            m_starts[m_label_count] = second();
            m_label_count++;

        } else {
            m_starts[idx] = second();
        }
    }

    double get_time(const char* label) const{
        return m_times[find_index(label)];
    }

    CUDA_HOSTDEV void stop(const char* label) {

        int idx = find_index(label);

        m_times[idx] += second() - m_starts[idx];
    }

    CUDA_HOSTDEV void print() const {

        for (size_t i = 0; i < m_label_count; ++i) {
            printf("%s : %f \n", m_labels[i].data(), m_times[i]);
        }
    }

private:
    size_t m_label_count;

    std::array<char_array, MAX_INSTANCES> m_labels;
    std::array<double, MAX_INSTANCES>     m_starts;
    std::array<double, MAX_INSTANCES>     m_times;

    CUDA_HOSTDEV int find_index(const char* label) const {

        for (size_t i = 0; i < m_labels.size(); ++i) {
            if (my_strcmp(label, m_labels[i].data()) == 0) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }
};