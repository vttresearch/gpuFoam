

#tests
g++ -pthread -std=c++17 -Iinclude/ -Itest/ test/test.cpp -o test_cpu.bin
#nvcc -expt-relaxed-constexpr -std=c++14 -Iinclude/ -Itest/ -x cu test/test.cpp -o test_gpu.bin
nvc++ -std=c++14 -Iinclude/ -Itest/ -x cu test/test.cpp -o test_gpu.bin


#nvcc -std=c++14 --default-stream per-thread -Iinclude/ -Itest/ -x cu test/benchmark_stream.cu -o benchmark_stream.bin

#benchmarks
#g++ -Iinclude/ -Itest/ test/benchmark.cpp -o benchmark_cpu.bin
#nvcc -Iinclude/ -Itest/ -x cu test/benchmark.cpp -o benchmark_gpu.bin
