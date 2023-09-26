
#include "error_handling.H"
#include "thermosAndReactions.H"

namespace FoamGpu {

template <class T>
static inline T* allocateAndTransfer(const std::vector<T>& t) {
    T*         ptr;
    const auto size     = t.size();
    const auto bytesize = size * sizeof(T);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&ptr, bytesize));
    CHECK_CUDA_ERROR(
        cudaMemcpy(ptr, t.data(), bytesize, cudaMemcpyHostToDevice));

    return ptr;
}

thermosAndReactions::thermosAndReactions
(
    const std::vector<gpuThermo>    thermos,
    const std::vector<gpuReaction>& reactions
)
    : nThermos_(gLabel(thermos.size()))
    , nReactions_(gLabel(reactions.size()))
    , thermos_(allocateAndTransfer(thermos))
    , reactions_(allocateAndTransfer(reactions)) {}


thermosAndReactions::~thermosAndReactions()
{
    this->deallocate();
}

void thermosAndReactions::allocate()
{
    CHECK_CUDA_ERROR
    (
        cudaMalloc((void**)&thermos_,nThermos_*sizeof(gpuThermo))
    ); 

    CHECK_CUDA_ERROR
    (
        cudaMalloc((void**)&reactions_,nReactions_*sizeof(gpuReaction))
    );

}
void thermosAndReactions::deallocate()
{
    CHECK_CUDA_ERROR(cudaFree(thermos_));
    CHECK_CUDA_ERROR(cudaFree(reactions_));
}


} // namespace FoamGpu