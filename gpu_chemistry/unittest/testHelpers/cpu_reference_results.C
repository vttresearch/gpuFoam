#include "cpu_reference_results.H"
#include "mock_of_odesystem.H"



namespace TestData{

std::vector<std::pair<gScalar, gScalar>> Thigh_Tlow_result(Mechanism mech)
{
    const Foam::ReactionList<FoamThermoType> reactions(
        TestData::makeSpeciesTable(mech),
        TestData::makeCpuThermos(mech),
        TestData::makeReactionDict(mech)
    );

    std::vector<std::pair<gScalar, gScalar>> ret;
    for (Foam::label i = 0; i < reactions.size(); ++i)
    {
        auto high = reactions[i].Thigh();
        auto low = reactions[i].Tlow();

        std::pair<gScalar, gScalar> p(high, low);
        ret.push_back(p);

    }

    return ret;

}

std::vector<gScalar> Kc_result(Mechanism mech)
{
    const Foam::ReactionList<FoamThermoType> reactions(
        TestData::makeSpeciesTable(mech),
        TestData::makeCpuThermos(mech),
        TestData::makeReactionDict(mech)
    );

    const gLabel nSpecie = TestData::speciesCount(mech);


    const Foam::scalar p = TestData::pInf(mech);
    const Foam::scalar T = TestData::TInf(mech);

    std::vector<gScalar> ret;
    for (Foam::label i = 0; i < reactions.size(); ++i)
    {
        gScalar Kc = reactions[i].Kc(p, T);
        ret.push_back(Kc);
    }
    return ret;

}


std::vector<gScalar> omega_result(Mechanism mech)
{
    const Foam::ReactionList<FoamThermoType> reactions(
        TestData::makeSpeciesTable(mech),
        TestData::makeCpuThermos(mech),
        TestData::makeReactionDict(mech)
    );

    const gLabel nSpecie = TestData::speciesCount(mech);

    Foam::scalarField c(nSpecie);
    assign_test_concentration(c, mech);
    Foam::scalar p = TestData::pInf(mech);
    Foam::scalar T = TestData::TInf(mech);

    std::vector<gScalar> ret;
    for (Foam::label i = 0; i < reactions.size(); ++i)
    {
        Foam::scalar omegaf = 0.3;
        Foam::scalar omegar = 0.4;
        const gLabel li = 0;

        ret.push_back(
            reactions[i].omega(p, T, c, li, omegaf, omegar)
        );

    }


    return ret;

}



std::vector<std::vector<gScalar>> dndtbyv_result(Mechanism mech){


    //Foam::MockOFSystem system(mech);
    //const auto& reactions = system.getReactions();

    const Foam::ReactionList<FoamThermoType> reactions(
        TestData::makeSpeciesTable(mech),
        TestData::makeCpuThermos(mech),
        TestData::makeReactionDict(mech)
    );


    const gLabel nSpecie = TestData::speciesCount(mech);

    //const gScalar p = 1E5;
    //const gScalar T = 431.4321;
    const gLabel li = 0;

    std::vector<std::vector<gScalar>> ret;
    for (Foam::label i = 0; i < reactions.size(); ++i)
    {


        Foam::scalarField c(nSpecie);
        assign_test_concentration(c, mech);


        Foam::scalar p = TestData::pInf(mech);
        Foam::scalar T = TestData::pInf(mech);

        Foam::scalarField result(c.size(), 0);

        reactions[i].dNdtByV(p, T, c, li, result, false, Foam::List<Foam::label>{}, 0);



        ret.push_back
        (
            std::vector<gScalar>(result.begin(), result.end())
        );


    }


    return ret;

}


std::vector<gScalar> derivative_result(Mechanism mech){

    const gLabel nEqns = TestData::equationCount(mech);
    Foam::MockOFSystem system(mech);

    const Foam::scalarField y0 = [=](){
        gLabel nEqns = TestData::equationCount(mech);
        Foam::scalarField y0_t(nEqns);
        assign_test_condition(y0_t, mech);
        return y0_t;
    }();

    const Foam::scalarField y = y0;
    Foam::scalarField dy(nEqns, 0.31);
    const gLabel li = 0;

    system.derivatives(0.0, y, li, dy);

    return std::vector<gScalar>(dy.begin(), dy.end());


}

}



