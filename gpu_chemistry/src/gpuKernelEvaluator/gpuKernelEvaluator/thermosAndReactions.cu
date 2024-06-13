#include "thermosAndReactions.H"

#include "device_allocate.H"
#include "device_free.H"
#include "error_handling.H"
#include "host_device_transfers.H"

namespace FoamGpu {

template <class T>
static inline T* allocateAndTransfer(const std::vector<T>& t) {

    T* ptr = device_allocate<T>(t.size());
    host_to_device(t.begin(), t.end(), ptr);
    return ptr;
}

thermosAndReactions::thermosAndReactions(
    const std::vector<gpuThermo>    thermos,
    const std::vector<gpuReaction>& reactions)
    : nThermos_(gLabel(thermos.size()))
    , nReactions_(gLabel(reactions.size()))
    , thermos_(allocateAndTransfer(thermos))
    , reactions_(allocateAndTransfer(reactions)) {}

thermosAndReactions::~thermosAndReactions() { this->deallocate(); }

void thermosAndReactions::deallocate() {
    device_free(thermos_);
    device_free(reactions_);
}

} // namespace FoamGpu