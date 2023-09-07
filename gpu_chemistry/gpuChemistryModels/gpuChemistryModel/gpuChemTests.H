#pragma once


#include "volFields.H"



namespace Foam{

    template<class Rcpu, class Rgpu>
    static void testThermo(const Rcpu& rcpu, const Rgpu& rgpu)
    {
        scalarField c(rcpu.size(), 0);
        c[0] = 0.1;
        c[1] = 0.3;
        c[2] = 0.2;
        const scalar T = 400;
        const scalar p = 1E5;

        bool ok = true;

        for (label i = 0; i < rcpu.size(); ++i)
        {
            auto cpu1 = rcpu[i].K(p, T);
            auto gpu1 = rgpu[i].K(p, T);
            if (std::abs(cpu1 - gpu1) > 1E-8){
                Info << "thermo::K() not working" << endl;
                ok = false;
                break;
            }
            ////
            auto cpu2 = rcpu[i].Kp(p, T);
            auto gpu2 = rgpu[i].Kp(p, T);

            if (std::abs(cpu2 - gpu2) > 1E-8){
                Info << "thermo::Kp() not working" << endl;
                ok = false;
                break;
            }
            ////
            auto cpu3 = rcpu[i].Kc(p, T);
            auto gpu3 = rgpu[i].Kc(p, T);

            if (std::abs(cpu3 - gpu3) > 1E-8){
                Info << "thermo::Kc() not working" << endl;
                ok = false;
                break;
            }
            ////
            auto cpu4 = rcpu[i].Y();
            auto gpu4 = rgpu[i].Y();

            if (std::abs(cpu4 - gpu4) > 1E-8){
                Info << "thermo::Y() not working" << endl;
                ok = false;
                break;
            }
            ////
            auto cpu5 = rcpu[i].W();
            auto gpu5 = rgpu[i].W();

            if (std::abs(cpu5 - gpu5) > 1E-8){
                Info << "thermo::W() not working" << endl;
                ok = false;
                break;
            }
        }

        if (ok){
            Info << "GpuThermos ok" << endl;
        }
    }


    template<class Rcpu, class Rgpu>
    static void testReactions(const Rcpu& rcpu, const Rgpu& rgpu)
    {
        scalarField c(rcpu.size(), 0);
        c[0] = 0.1;
        c[1] = 0.3;
        c[2] = 0.2;
        const scalar T = 400;
        const scalar p = 1E5;
        const label li = 0;

        bool ok = true;

        //Tests for Reaction::C()
        {
            scalar cf = 1000;
            scalar cr = 2000;

            for (label i = 0; i < rcpu.size(); ++i)
            {

                rgpu[i].C(p, T, c, li, cf, cr);
                auto o_gpu = cf + cr;
                cf = 1000;
                cr = 2000;
                rcpu[i].C(p, T, c, li, cf, cr);
                auto o_cpu = cf + cr;

                if (std::abs(o_gpu - o_cpu) > 1E-8)
                {
                    Info << "Reaction::C() not working" << endl;
                    ok = false;
                    break;
                }
            }
        }

        //Tests for Reaction::Y()
        {

            for (label i = 0; i < rcpu.size(); ++i)
            {

                auto o_gpu = rgpu[i].Y();
                auto o_cpu = rcpu[i].Y();

                if (std::abs(o_gpu - o_cpu) > 1E-8)
                {
                    Info << "Reaction::Y() not working: " << o_gpu << " " << o_cpu << endl;
                    ok = false;
                    break;
                }
            }
        }

        //Tests for Reaction::W()
        {

            for (label i = 0; i < rcpu.size(); ++i)
            {

                auto o_gpu = rgpu[i].W();
                auto o_cpu = rcpu[i].W();

                if (std::abs(o_gpu - o_cpu) > 1E-8)
                {
                    Info << "Reaction::W() not working: " << o_gpu << " " << o_cpu << endl;
                    ok = false;
                    break;
                }
            }
        }

        //Tests for Reaction::Kc()
        {

            for (label i = 0; i < rcpu.size(); ++i)
            {

                auto o_gpu = rgpu[i].Kc(p, T);
                auto o_cpu = rcpu[i].Kc(p, T);

                if (std::abs(o_gpu - o_cpu) > 1E-8)
                {
                    Info << "Reaction::Kc() not working: " << o_gpu << " " << o_cpu << endl;
                    ok = false;
                    break;
                }
            }
        }

        //Tests for Reaction::kf()
        {
            for (label i = 0; i < rcpu.size(); ++i)
            {

                auto o_gpu = rgpu[i].kf(p, T, c, li);
                auto o_cpu = rcpu[i].kf(p, T, c, li);

                if (std::abs(o_gpu - o_cpu) > 1E-8)
                {
                    Info << "Reaction::kf() not working: " << o_gpu << " " << o_cpu << endl;
                    ok = false;
                    break;
                }
            }
        }


        //Tests for Reaction::kr()
        {
            for (label i = 0; i < rcpu.size(); ++i)
            {

                auto o_gpu = rgpu[i].kr(p, T, c, li);
                auto o_cpu = rcpu[i].kr(p, T, c, li);

                if (std::abs(o_gpu - o_cpu) > 1E-8)
                {
                    Info << "Reaction::kr() not working: " << o_gpu << " " << o_cpu << endl;
                    ok = false;
                    break;
                }
            }
        }

        if (ok) {
            Info << "gpuReactions ok" << endl;
        }

    }


}

