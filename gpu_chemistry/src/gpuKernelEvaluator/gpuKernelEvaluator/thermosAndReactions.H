#pragma once

#include <vector>

#include "gpuReaction.H"
#include "gpuThermo.H"


namespace FoamGpu{

struct thermosAndReactions{

    thermosAndReactions() = default;

    thermosAndReactions
    (
        const std::vector<gpuThermo> thermos,
        const std::vector<gpuReaction>& reactions
    );

    ~thermosAndReactions();


    gpuThermo* thermos() {return thermos_;}
    gpuReaction* reactions() {return reactions_;}


private:
    gLabel nThermos_;
    gLabel nReactions_;
    gpuThermo* thermos_;
    gpuReaction* reactions_;


    void allocate();
    void deallocate();


};

}