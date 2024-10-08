R"(reactions
{
    un-named-reaction-0
    {
        type            reversibleArrheniusReaction;
        reaction        "H + O2 = OH + O";
        A               9.756e+10;
        beta            0;
        Ta              7468.5;
    }
    un-named-reaction-1
    {
        type            reversibleArrheniusReaction;
        reaction        "O + H2 = H + OH";
        A               45.89;
        beta            2.7;
        Ta              3149.98;
    }
    un-named-reaction-2
    {
        type            reversibleArrheniusReaction;
        reaction        "OH + H2 = H2O + H";
        A               102400;
        beta            1.6;
        Ta              1659.67;
    }
    un-named-reaction-3
    {
        type            reversibleArrheniusReaction;
        reaction        "OH + OH = O + H2O";
        A               39.73;
        beta            2.4;
        Ta              -1061.73;
    }
    un-named-reaction-4
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "H + O2 = HO2";
        k0
        {
            A               6.328e+13;
            beta            -1.4;
            Ta              0;
        }
        kInf
        {
            A               5.116e+09;
            beta            0.44;
            Ta              0;
        }
        F
        {
            alpha           0.5;
            Tsss            1e-30;
            Ts              1e+30;
            Tss             1e+15;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.4)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 1)
(H2O 11.89)
(H2O2 1)
(O2 0.85)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 1)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.09)
(CO2 2.18)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 1)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-5
    {
        type            reversibleArrheniusReaction;
        reaction        "HO2 + H = OH + OH";
        A               7.485e+10;
        beta            0;
        Ta              148.441;
    }
    un-named-reaction-6
    {
        type            reversibleArrheniusReaction;
        reaction        "H2 + O2 = HO2 + H";
        A               591.6;
        beta            2.433;
        Ta              26921.7;
    }
    un-named-reaction-7
    {
        type            reversibleArrheniusReaction;
        reaction        "HO2 + OH = H2O + O2";
        A               2.891e+10;
        beta            0;
        Ta              -252.557;
    }
    un-named-reaction-8
    {
        type            reversibleArrheniusReaction;
        reaction        "HO2 + H = O + H2O";
        A               3.97e+09;
        beta            0;
        Ta              337.641;
    }
    un-named-reaction-9
    {
        type            reversibleArrheniusReaction;
        reaction        "HO2 + O = OH + O2";
        A               4e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-10
    {
        type            reversibleArrheniusReaction;
        reaction        "HO2 + HO2 = O2 + H2O2";
        A               1.3e+08;
        beta            0;
        Ta              -820.202;
    }
    un-named-reaction-11
    {
        type            reversibleArrheniusReaction;
        reaction        "HO2 + HO2 = O2 + H2O2";
        A               3.658e+11;
        beta            0;
        Ta              6038.29;
    }
    un-named-reaction-12
    {
        type            reversibleArrheniusReaction;
        reaction        "H2O2 + H = OH + H2O";
        A               2.41e+10;
        beta            0;
        Ta              1997.67;
    }
    un-named-reaction-13
    {
        type            reversibleArrheniusReaction;
        reaction        "H2O2 + H = HO2 + H2";
        A               6050;
        beta            2;
        Ta              2616.59;
    }
    un-named-reaction-14
    {
        type            reversibleArrheniusReaction;
        reaction        "H2O2 + O = OH + HO2";
        A               9630;
        beta            2;
        Ta              1997.67;
    }
    un-named-reaction-15
    {
        type            reversibleArrheniusReaction;
        reaction        "H2O2 + OH = HO2 + H2O";
        A               2e+09;
        beta            0;
        Ta              214.863;
    }
    un-named-reaction-16
    {
        type            reversibleArrheniusReaction;
        reaction        "H2O2 + OH = HO2 + H2O";
        A               2.67e+38;
        beta            -7;
        Ta              18920;
    }
    un-named-reaction-17
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "OH + OH = H2O2";
        k0
        {
            A               2.01e+11;
            beta            -0.584;
            Ta              -1153.82;
        }
        kInf
        {
            A               1.11e+11;
            beta            -0.37;
            Ta              0;
        }
        F
        {
            alpha           0.7346;
            Tsss            94;
            Ts              1756;
            Tss             5182;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 1)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.75)
(CO2 3.6)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 1)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-18
    {
        type            reversibleThirdBodyArrhenius;
        reaction        "H + H = H2";
        A               1.78e+12;
        beta            -1;
        Ta              0;
        coeffs
54
(
(N2 1)
(AR 0.63)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 0)
(H2O 0)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 1)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1)
(CO2 0)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 1)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
    }
    un-named-reaction-19
    {
        type            reversibleThirdBodyArrhenius;
        reaction        "H + OH = H2O";
        A               4.4e+16;
        beta            -2;
        Ta              0;
        coeffs
54
(
(N2 1)
(AR 0.38)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6.3)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 1)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.75)
(CO2 3.6)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 1)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
    }
    un-named-reaction-20
    {
        type            reversibleThirdBodyArrhenius;
        reaction        "O + O = O2";
        A               1.2e+11;
        beta            -1;
        Ta              0;
        coeffs
54
(
(N2 1)
(AR 0.83)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2.4)
(H2O 15.4)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 1)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.75)
(CO2 3.6)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 1)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
    }
    un-named-reaction-21
    {
        type            reversibleArrheniusReaction;
        reaction        "H + H + H2 = H2 + H2";
        A               9e+10;
        beta            -0.6;
        Ta              0;
    }
    un-named-reaction-22
    {
        type            reversibleArrheniusReaction;
        reaction        "H + H + H2O = H2 + H2O";
        A               5.624e+13;
        beta            -1.25;
        Ta              0;
    }
    un-named-reaction-23
    {
        type            reversibleArrheniusReaction;
        reaction        "H + H + CO2 = H2 + CO2";
        A               5.5e+14;
        beta            -2;
        Ta              0;
    }
    un-named-reaction-24
    {
        type            reversibleThirdBodyArrhenius;
        reaction        "O + H = OH";
        A               9.428e+12;
        beta            -1;
        Ta              0;
        coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 12)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 1)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.75)
(CO2 3.6)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 1)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
    }
    un-named-reaction-25
    {
        type            reversibleArrheniusReaction;
        reaction        "CO + OH = CO2 + H";
        A               70.46;
        beta            2.053;
        Ta              -178.97;
    }
    un-named-reaction-26
    {
        type            reversibleArrheniusReaction;
        reaction        "CO + OH = CO2 + H";
        A               5.757e+09;
        beta            -0.664;
        Ta              166.974;
    }
    un-named-reaction-27
    {
        type            reversibleArrheniusReaction;
        reaction        "CO + HO2 = CO2 + OH";
        A               157;
        beta            2.18;
        Ta              9028.56;
    }
    un-named-reaction-28
    {
        type            reversibleArrheniusLindemannFallOffReaction;
        reaction        "CO + O = CO2";
        k0
        {
            A               1.173e+18;
            beta            -2.79;
            Ta              2108.87;
        }
        kInf
        {
            A               1.362e+07;
            beta            0;
            Ta              1199.61;
        }
        F
        {
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 12)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 1)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.75)
(CO2 3.6)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 1)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-29
    {
        type            reversibleArrheniusReaction;
        reaction        "CO + O2 = CO2 + O";
        A               1.119e+09;
        beta            0;
        Ta              24002.2;
    }
    un-named-reaction-30
    {
        type            reversibleThirdBodyArrhenius;
        reaction        "HCO = CO + H";
        A               1.87e+14;
        beta            -1;
        Ta              8554.25;
        coeffs
54
(
(N2 1)
(AR 1)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 0)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 1)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.75)
(CO2 3.6)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 1)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
    }
    un-named-reaction-31
    {
        type            reversibleArrheniusReaction;
        reaction        "HCO + H = CO + H2";
        A               1.2e+11;
        beta            0;
        Ta              0;
    }
    un-named-reaction-32
    {
        type            reversibleArrheniusReaction;
        reaction        "HCO + O = CO + OH";
        A               3e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-33
    {
        type            reversibleArrheniusReaction;
        reaction        "HCO + O = CO2 + H";
        A               3e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-34
    {
        type            reversibleArrheniusReaction;
        reaction        "HCO + OH = CO + H2O";
        A               3.02e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-35
    {
        type            reversibleArrheniusReaction;
        reaction        "HCO + O2 = CO + HO2";
        A               1.204e+07;
        beta            0.807;
        Ta              -365.82;
    }
    un-named-reaction-36
    {
        type            reversibleArrheniusReaction;
        reaction        "HCO + H2O = CO + H + H2O";
        A               2.244e+15;
        beta            -1;
        Ta              8554.25;
    }
    un-named-reaction-37
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "CO + H2 = CH2O";
        k0
        {
            A               5.07e+21;
            beta            -3.42;
            Ta              42444.2;
        }
        kInf
        {
            A               43000;
            beta            1.5;
            Ta              40054;
        }
        F
        {
            alpha           0.932;
            Tsss            197;
            Ts              1540;
            Tss             10300;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-38
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "HCO + H = CH2O";
        k0
        {
            A               1.35e+18;
            beta            -2.57;
            Ta              717.048;
        }
        kInf
        {
            A               1.09e+09;
            beta            0.48;
            Ta              -130.83;
        }
        F
        {
            alpha           0.7824;
            Tsss            271;
            Ts              2755;
            Tss             6570;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-39
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "CH2 + H = CH3";
        k0
        {
            A               3.2e+21;
            beta            -3.14;
            Ta              618.925;
        }
        kInf
        {
            A               2.5e+13;
            beta            -0.8;
            Ta              0;
        }
        F
        {
            alpha           0.68;
            Tsss            78;
            Ts              1995;
            Tss             5590;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-40
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2 + O = HCO + H";
        A               8e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-41
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2 + OH = CH2O + H";
        A               2e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-42
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2 + H2 = H + CH3";
        A               500;
        beta            2;
        Ta              3638.07;
    }
    un-named-reaction-43
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2 + O2 = HCO + OH";
        A               1.06e+10;
        beta            0;
        Ta              754.787;
    }
    un-named-reaction-44
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2 + O2 = CO2 + H + H";
        A               2.64e+09;
        beta            0;
        Ta              754.787;
    }
    un-named-reaction-45
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2 + HO2 = CH2O + OH";
        A               2e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-46
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2 + CH2 = C2H2 + H2";
        A               3.2e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-47
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + N2 = CH2 + N2";
        A               1.5e+10;
        beta            0;
        Ta              301.915;
    }
    un-named-reaction-48
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + AR = CH2 + AR";
        A               9e+09;
        beta            0;
        Ta              301.915;
    }
    un-named-reaction-49
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + O = CO + H2";
        A               1.5e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-50
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + O = HCO + H";
        A               1.5e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-51
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + OH = CH2O + H";
        A               3e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-52
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + H2 = CH3 + H";
        A               7e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-53
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + O2 = H + OH + CO";
        A               2.8e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-54
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + O2 = CO + H2O";
        A               1.2e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-55
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + H2O = CH2 + H2O";
        A               3e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-56
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + CO = CH2 + CO";
        A               9e+09;
        beta            0;
        Ta              0;
    }
    un-named-reaction-57
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + CO2 = CH2 + CO2";
        A               7e+09;
        beta            0;
        Ta              0;
    }
    un-named-reaction-58
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2* + CO2 = CH2O + CO";
        A               1.4e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-59
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "CH2O + H = CH3O";
        k0
        {
            A               2.2e+24;
            beta            -4.8;
            Ta              2797.74;
        }
        kInf
        {
            A               5.4e+08;
            beta            0.454;
            Ta              1308.3;
        }
        F
        {
            alpha           0.758;
            Tsss            94;
            Ts              1555;
            Tss             4200;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 1)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-60
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2O + H = HCO + H2";
        A               2.3e+07;
        beta            1.05;
        Ta              1647.95;
    }
    un-named-reaction-61
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2O + O = HCO + OH";
        A               3.9e+10;
        beta            0;
        Ta              1781.3;
    }
    un-named-reaction-62
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2O + OH = HCO + H2O";
        A               3.43e+06;
        beta            1.18;
        Ta              -224.926;
    }
    un-named-reaction-63
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2O + O2 = HCO + HO2";
        A               1e+11;
        beta            0;
        Ta              20127.6;
    }
    un-named-reaction-64
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2O + HO2 = HCO + H2O2";
        A               1e+09;
        beta            0;
        Ta              4025.53;
    }
    un-named-reaction-65
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "CH3 + H = CH4";
        k0
        {
            A               2.477e+27;
            beta            -4.76;
            Ta              1227.79;
        }
        kInf
        {
            A               1.27e+13;
            beta            -0.63;
            Ta              192.722;
        }
        F
        {
            alpha           0.783;
            Tsss            74;
            Ts              2941;
            Tss             6964;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-66
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + O = CH2O + H";
        A               8.43e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-67
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + OH = CH2 + H2O";
        A               56000;
        beta            1.6;
        Ta              2727.3;
    }
    un-named-reaction-68
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + OH = CH2* + H2O";
        A               2.501e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-69
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + O2 = O + CH3O";
        A               3.083e+10;
        beta            0;
        Ta              14491.9;
    }
    un-named-reaction-70
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + O2 = OH + CH2O";
        A               3.6e+07;
        beta            0;
        Ta              4498.53;
    }
    un-named-reaction-71
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + HO2 = CH4 + O2";
        A               1e+09;
        beta            0;
        Ta              0;
    }
    un-named-reaction-72
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + HO2 = CH3O + OH";
        A               1.34e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-73
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + H2O2 = CH4 + HO2";
        A               24.5;
        beta            2.47;
        Ta              2606.53;
    }
    un-named-reaction-74
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + HCO = CH4 + CO";
        A               8.48e+09;
        beta            0;
        Ta              0;
    }
    un-named-reaction-75
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + CH2O = CH4 + HCO";
        A               3.32;
        beta            2.81;
        Ta              2948.7;
    }
    un-named-reaction-76
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + CH2 = C2H4 + H";
        A               4e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-77
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + CH2* = C2H4 + H";
        A               1.2e+10;
        beta            0;
        Ta              -286.819;
    }
    un-named-reaction-78
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "CH3 + CH3 = C2H6";
        k0
        {
            A               1.77e+44;
            beta            -9.67;
            Ta              3129.85;
        }
        kInf
        {
            A               2.12e+13;
            beta            -0.97;
            Ta              311.979;
        }
        F
        {
            alpha           0.5325;
            Tsss            151;
            Ts              1038;
            Tss             4970;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-79
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3 + CH3 = H + C2H5";
        A               4.99e+09;
        beta            0.1;
        Ta              5333.83;
    }
    un-named-reaction-80
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3O + H = CH2O + H2";
        A               2e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-81
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3O + H = CH3 + OH";
        A               3.2e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-82
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3O + H = CH2* + H2O";
        A               1.6e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-83
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3O + O = CH2O + OH";
        A               1e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-84
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3O + OH = CH2O + H2O";
        A               5e+09;
        beta            0;
        Ta              0;
    }
    un-named-reaction-85
    {
        type            reversibleArrheniusReaction;
        reaction        "CH3O + O2 = CH2O + HO2";
        A               4.28e-16;
        beta            7.6;
        Ta              -1776.27;
    }
    un-named-reaction-86
    {
        type            reversibleArrheniusReaction;
        reaction        "CH4 + H = CH3 + H2";
        A               660000;
        beta            1.62;
        Ta              5454.59;
    }
    un-named-reaction-87
    {
        type            reversibleArrheniusReaction;
        reaction        "CH4 + O = CH3 + OH";
        A               1.02e+06;
        beta            1.5;
        Ta              4327.44;
    }
    un-named-reaction-88
    {
        type            reversibleArrheniusReaction;
        reaction        "CH4 + OH = CH3 + H2O";
        A               100000;
        beta            1.6;
        Ta              1569.96;
    }
    un-named-reaction-89
    {
        type            reversibleArrheniusReaction;
        reaction        "CH4 + CH2 = CH3 + CH3";
        A               2460;
        beta            2;
        Ta              4161.39;
    }
    un-named-reaction-90
    {
        type            reversibleArrheniusReaction;
        reaction        "CH4 + CH2* = CH3 + CH3";
        A               1.6e+10;
        beta            0;
        Ta              -286.819;
    }
    un-named-reaction-91
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C2H3 = C2H2 + H";
        k0
        {
            A               2.565e+24;
            beta            -3.4;
            Ta              18013.6;
        }
        kInf
        {
            A               3.86e+07;
            beta            1.62;
            Ta              18642.3;
        }
        F
        {
            alpha           1.9816;
            Tsss            5383.7;
            Ts              4.2932;
            Tss             -0.0795;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 3)
(C2H3 1)
(C2H4 3)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-92
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H2 + O = CH2 + CO";
        A               4080;
        beta            2;
        Ta              956.063;
    }
    un-named-reaction-93
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H2 + OH = CH3 + CO";
        A               4.83e-07;
        beta            4;
        Ta              -1006.38;
    }
    un-named-reaction-94
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H2 + HCO = C2H3 + CO";
        A               10000;
        beta            2;
        Ta              3019.15;
    }
    un-named-reaction-95
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H2 + CH3 = aC3H5";
        A               2.68e+50;
        beta            -12.82;
        Ta              17979;
    }
    un-named-reaction-96
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C2H3 + H = C2H4";
        k0
        {
            A               1.4e+24;
            beta            -3.86;
            Ta              1670.59;
        }
        kInf
        {
            A               6.08e+09;
            beta            0.27;
            Ta              140.894;
        }
        F
        {
            alpha           0.782;
            Tsss            207.5;
            Ts              2663;
            Tss             6095;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 3)
(C2H3 1)
(C2H4 3)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-97
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + H = C2H2 + H2";
        A               9e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-98
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + O = CH3 + CO";
        A               4.8e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-99
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + OH = C2H2 + H2O";
        A               3.011e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-100
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + O2 = C2H2 + HO2";
        A               1340;
        beta            1.61;
        Ta              -192.924;
    }
    un-named-reaction-101
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + O2 = CH2CHO + O";
        A               3e+08;
        beta            0.29;
        Ta              5.5351;
    }
    un-named-reaction-102
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + O2 = HCO + CH2O";
        A               4.6e+13;
        beta            -1.39;
        Ta              508.223;
    }
    un-named-reaction-103
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + HO2 = CH2CHO + OH";
        A               1e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-104
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + H2O2 = C2H4 + HO2";
        A               1.21e+07;
        beta            0;
        Ta              -299.902;
    }
    un-named-reaction-105
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + HCO = C2H4 + CO";
        A               9.033e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-106
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + HCO = C2H3CHO";
        A               1.8e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-107
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + CH3 = C2H2 + CH4";
        A               3.92e+08;
        beta            0;
        Ta              0;
    }
    un-named-reaction-108
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C2H3 + CH3 = C3H6";
        k0
        {
            A               4.27e+52;
            beta            -11.94;
            Ta              4916.08;
        }
        kInf
        {
            A               2.5e+10;
            beta            0;
            Ta              0;
        }
        F
        {
            alpha           0.175;
            Tsss            1340.6;
            Ts              60000;
            Tss             10139.8;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 3)
(C2H3 1)
(C2H4 3)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-109
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + CH3 = aC3H5 + H";
        A               1.5e+21;
        beta            -2.83;
        Ta              9368.41;
    }
    un-named-reaction-110
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3 + C2H3 = C2H2 + C2H4";
        A               9.6e+08;
        beta            0;
        Ta              0;
    }
    un-named-reaction-111
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2CHO = CH3 + CO";
        A               7.8e+41;
        beta            -9.147;
        Ta              23599.7;
    }
    un-named-reaction-112
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2CHO + H = CH3 + HCO";
        A               9e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-113
    {
        type            reversibleArrheniusReaction;
        reaction        "CH2CHO + O2 = CH2O + CO + OH";
        A               1.8e+07;
        beta            0;
        Ta              0;
    }
    un-named-reaction-114
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C2H4 + H = C2H5";
        k0
        {
            A               4.715e+12;
            beta            0;
            Ta              380.04;
        }
        kInf
        {
            A               3.975e+06;
            beta            1.28;
            Ta              649.434;
        }
        F
        {
            alpha           0.76;
            Tsss            40;
            Ts              1025;
            Tss             1e+15;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-115
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + H = C2H3 + H2";
        A               50700;
        beta            1.9;
        Ta              6516.33;
    }
    un-named-reaction-116
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + O = C2H3 + OH";
        A               15100;
        beta            1.9;
        Ta              1881.94;
    }
    un-named-reaction-117
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + O = CH3 + HCO";
        A               19200;
        beta            1.83;
        Ta              110.702;
    }
    un-named-reaction-118
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + O = CH2 + CH2O";
        A               384;
        beta            1.83;
        Ta              110.702;
    }
    un-named-reaction-119
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + OH = C2H3 + H2O";
        A               3600;
        beta            2;
        Ta              1257.98;
    }
    un-named-reaction-120
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + HCO = C2H5 + CO";
        A               10000;
        beta            2;
        Ta              4025.53;
    }
    un-named-reaction-121
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + CH2 = aC3H5 + H";
        A               2e+10;
        beta            0;
        Ta              3019.15;
    }
    un-named-reaction-122
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + CH2* = aC3H5 + H";
        A               5e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-123
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + CH3 = C2H3 + CH4";
        A               227;
        beta            2;
        Ta              4629.36;
    }
    un-named-reaction-124
    {
        type            reversibleArrheniusReaction;
        reaction        "nC3H7 = CH3 + C2H4";
        A               9.6e+13;
        beta            0;
        Ta              15610.5;
    }
    un-named-reaction-125
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + O2 = C2H3 + HO2";
        A               4.22e+10;
        beta            0;
        Ta              30594;
    }
    un-named-reaction-126
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + C2H3 = C4H7";
        A               7.93e+35;
        beta            -8.47;
        Ta              7155.38;
    }
    un-named-reaction-127
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C2H5 + H = C2H6";
        k0
        {
            A               1.99e+35;
            beta            -7.08;
            Ta              3363.83;
        }
        kInf
        {
            A               5.21e+14;
            beta            -0.99;
            Ta              795.042;
        }
        F
        {
            alpha           0.8422;
            Tsss            125;
            Ts              2219;
            Tss             6882;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-128
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H5 + H = C2H4 + H2";
        A               2e+09;
        beta            0;
        Ta              0;
    }
    un-named-reaction-129
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H5 + O = CH3 + CH2O";
        A               1.604e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-130
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H5 + O2 = C2H4 + HO2";
        A               2e+07;
        beta            0;
        Ta              0;
    }
    un-named-reaction-131
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H5 + HO2 = C2H6 + O2";
        A               3e+08;
        beta            0;
        Ta              0;
    }
    un-named-reaction-132
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H5 + HO2 = C2H4 + H2O2";
        A               3e+08;
        beta            0;
        Ta              0;
    }
    un-named-reaction-133
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H5 + HO2 = CH3 + CH2O + OH";
        A               2.4e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-134
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H5 + H2O2 = C2H6 + HO2";
        A               8.7e+06;
        beta            0;
        Ta              490.108;
    }
    un-named-reaction-135
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C2H5 + C2H3 = C4H81";
        k0
        {
            A               1.55e+50;
            beta            -11.79;
            Ta              4520.92;
        }
        kInf
        {
            A               1.5e+10;
            beta            0;
            Ta              0;
        }
        F
        {
            alpha           0.198;
            Tsss            2277.9;
            Ts              60000;
            Tss             5723.2;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-136
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H5 + C2H3 = aC3H5 + CH3";
        A               3.9e+29;
        beta            -5.22;
        Ta              9936.52;
    }
    un-named-reaction-137
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H6 + H = C2H5 + H2";
        A               115000;
        beta            1.9;
        Ta              3789.03;
    }
    un-named-reaction-138
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H6 + O = C2H5 + OH";
        A               89800;
        beta            1.92;
        Ta              2863.16;
    }
    un-named-reaction-139
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H6 + OH = C2H5 + H2O";
        A               3540;
        beta            2.12;
        Ta              437.776;
    }
    un-named-reaction-140
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H6 + CH2* = C2H5 + CH3";
        A               4e+10;
        beta            0;
        Ta              -276.755;
    }
    un-named-reaction-141
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H6 + CH3 = C2H5 + CH4";
        A               6140;
        beta            1.74;
        Ta              5258.35;
    }
    un-named-reaction-142
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "aC3H5 + H = C3H6";
        k0
        {
            A               1.33e+54;
            beta            -12;
            Ta              3002.94;
        }
        kInf
        {
            A               2e+11;
            beta            0;
            Ta              0;
        }
        F
        {
            alpha           0.02;
            Tsss            1096.6;
            Ts              1096.6;
            Tss             6859.5;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-143
    {
        type            reversibleArrheniusReaction;
        reaction        "aC3H5 + O = C2H3CHO + H";
        A               6e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-144
    {
        type            reversibleArrheniusReaction;
        reaction        "aC3H5 + OH = C2H3CHO + H + H";
        A               4.2e+29;
        beta            -5.16;
        Ta              15159.1;
    }
    un-named-reaction-145
    {
        type            reversibleArrheniusReaction;
        reaction        "aC3H5 + O2 = C2H3CHO + OH";
        A               1.82e+10;
        beta            -0.41;
        Ta              11502.4;
    }
    un-named-reaction-146
    {
        type            reversibleArrheniusReaction;
        reaction        "aC3H5 + HO2 = C3H6 + O2";
        A               2.66e+09;
        beta            0;
        Ta              0;
    }
    un-named-reaction-147
    {
        type            reversibleArrheniusReaction;
        reaction        "aC3H5 + HO2 = OH + C2H3 + CH2O";
        A               6.6e+09;
        beta            0;
        Ta              0;
    }
    un-named-reaction-148
    {
        type            reversibleArrheniusReaction;
        reaction        "aC3H5 + HCO = C3H6 + CO";
        A               6e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-149
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "aC3H5 + CH3 = C4H81";
        k0
        {
            A               3.91e+54;
            beta            -12.81;
            Ta              3144.95;
        }
        kInf
        {
            A               1e+11;
            beta            -0.32;
            Ta              -131.987;
        }
        F
        {
            alpha           0.104;
            Tsss            1606;
            Ts              60000;
            Tss             6118.4;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-150
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C3H6 + H = nC3H7";
        k0
        {
            A               6.26e+32;
            beta            -6.66;
            Ta              3522.34;
        }
        kInf
        {
            A               1.33e+10;
            beta            0;
            Ta              1640.76;
        }
        F
        {
            alpha           1;
            Tsss            1000;
            Ts              1310;
            Tss             48097;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-151
    {
        type            reversibleArrheniusReaction;
        reaction        "C3H6 + H = C2H4 + CH3";
        A               8e+18;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-152
    {
        type            reversibleArrheniusReaction;
        reaction        "C3H6 + H = aC3H5 + H2";
        A               173;
        beta            2.5;
        Ta              1252.95;
    }
    un-named-reaction-153
    {
        type            reversibleArrheniusReaction;
        reaction        "C3H6 + O = C2H3CHO + H + H";
        A               40000;
        beta            1.65;
        Ta              164.544;
    }
    un-named-reaction-154
    {
        type            reversibleArrheniusReaction;
        reaction        "C3H6 + O = C2H5 + HCO";
        A               35000;
        beta            1.65;
        Ta              -489.102;
    }
    un-named-reaction-155
    {
        type            reversibleArrheniusReaction;
        reaction        "C3H6 + O = aC3H5 + OH";
        A               1.8e+08;
        beta            0.7;
        Ta              2958.76;
    }
    un-named-reaction-156
    {
        type            reversibleArrheniusReaction;
        reaction        "C3H6 + OH = aC3H5 + H2O";
        A               3100;
        beta            2;
        Ta              -149.951;
    }
    un-named-reaction-157
    {
        type            reversibleArrheniusReaction;
        reaction        "C3H6 + HO2 = aC3H5 + H2O2";
        A               9.6;
        beta            2.6;
        Ta              6999.39;
    }
    un-named-reaction-158
    {
        type            reversibleArrheniusReaction;
        reaction        "C3H6 + CH3 = aC3H5 + CH4";
        A               0.0022;
        beta            3.5;
        Ta              2855.61;
    }
    un-named-reaction-159
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3CHO + H = C2H4 + HCO";
        A               1.08e+08;
        beta            0.454;
        Ta              2928.57;
    }
    un-named-reaction-160
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3CHO + O = C2H3 + OH + CO";
        A               3e+10;
        beta            0;
        Ta              1781.3;
    }
    un-named-reaction-161
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H3CHO + OH = C2H3 + H2O + CO";
        A               3.43e+06;
        beta            1.18;
        Ta              -224.926;
    }
    un-named-reaction-162
    {
        type            reversibleArrheniusReaction;
        reaction        "nC3H7 + H = C2H5 + CH3";
        A               3.7e+21;
        beta            -2.92;
        Ta              6292.41;
    }
    un-named-reaction-163
    {
        type            reversibleArrheniusReaction;
        reaction        "nC3H7 + H = C3H6 + H2";
        A               1.8e+09;
        beta            0;
        Ta              0;
    }
    un-named-reaction-164
    {
        type            reversibleArrheniusReaction;
        reaction        "nC3H7 + O = C2H5 + CH2O";
        A               9.6e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-165
    {
        type            reversibleArrheniusReaction;
        reaction        "nC3H7 + OH = C3H6 + H2O";
        A               2.4e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-166
    {
        type            reversibleArrheniusReaction;
        reaction        "nC3H7 + O2 = C3H6 + HO2";
        A               9e+07;
        beta            0;
        Ta              0;
    }
    un-named-reaction-167
    {
        type            reversibleArrheniusReaction;
        reaction        "nC3H7 + HO2 = C2H5 + OH + CH2O";
        A               2.4e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-168
    {
        type            reversibleArrheniusReaction;
        reaction        "nC3H7 + CH3 = CH4 + C3H6";
        A               1.1e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-169
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C4H7 + H = C4H81";
        k0
        {
            A               3.01e+42;
            beta            -9.32;
            Ta              2935.42;
        }
        kInf
        {
            A               3.6e+10;
            beta            0;
            Ta              0;
        }
        F
        {
            alpha           0.498;
            Tsss            1314;
            Ts              1314;
            Tss             50000;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-170
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H7 + H = CH3 + aC3H5";
        A               2e+18;
        beta            -2;
        Ta              5535.1;
    }
    un-named-reaction-171
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H7 + HO2 = CH2O + OH + aC3H5";
        A               2.4e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-172
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H7 + HCO = C4H81 + CO";
        A               6e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-173
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C4H81 + H = pC4H9";
        k0
        {
            A               6.26e+32;
            beta            -6.66;
            Ta              3522.34;
        }
        kInf
        {
            A               1.33e+10;
            beta            0;
            Ta              1640.76;
        }
        F
        {
            alpha           1;
            Tsss            1000;
            Ts              1310;
            Tss             48097;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-174
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + H = C2H4 + C2H5";
        A               1.6e+19;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-175
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + H = C3H6 + CH3";
        A               3.2e+19;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-176
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + H = C4H7 + H2";
        A               650;
        beta            2.54;
        Ta              3399.56;
    }
    un-named-reaction-177
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + O = nC3H7 + HCO";
        A               330000;
        beta            1.45;
        Ta              -202.283;
    }
    un-named-reaction-178
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + O = C4H7 + OH";
        A               1.5e+10;
        beta            0;
        Ta              2898.38;
    }
    un-named-reaction-179
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + O = C4H7 + OH";
        A               2.6e+10;
        beta            0;
        Ta              2249.26;
    }
    un-named-reaction-180
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + OH = C4H7 + H2O";
        A               0.7;
        beta            2.66;
        Ta              265.182;
    }
    un-named-reaction-181
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + O2 = C4H7 + HO2";
        A               2e+10;
        beta            0;
        Ta              25627.5;
    }
    un-named-reaction-182
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + HO2 = C4H7 + H2O2";
        A               1e+09;
        beta            0;
        Ta              7215.76;
    }
    un-named-reaction-183
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + CH3 = C4H7 + CH4";
        A               0.00045;
        beta            3.65;
        Ta              3599.33;
    }
    un-named-reaction-184
    {
        type            reversibleArrheniusReaction;
        reaction        "pC4H9 + H = C2H5 + C2H5";
        A               3.7e+21;
        beta            -2.92;
        Ta              6292.41;
    }
    un-named-reaction-185
    {
        type            reversibleArrheniusReaction;
        reaction        "pC4H9 + H = C4H81 + H2";
        A               1.8e+09;
        beta            0;
        Ta              0;
    }
    un-named-reaction-186
    {
        type            reversibleArrheniusReaction;
        reaction        "pC4H9 + O = nC3H7 + CH2O";
        A               9.6e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-187
    {
        type            reversibleArrheniusReaction;
        reaction        "pC4H9 + OH = C4H81 + H2O";
        A               2.4e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-188
    {
        type            reversibleArrheniusReaction;
        reaction        "pC4H9 + O2 = C4H81 + HO2";
        A               2.7e+08;
        beta            0;
        Ta              0;
    }
    un-named-reaction-189
    {
        type            reversibleArrheniusReaction;
        reaction        "pC4H9 + HO2 = nC3H7 + OH + CH2O";
        A               2.4e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-190
    {
        type            reversibleArrheniusReaction;
        reaction        "pC4H9 + CH3 = C4H81 + CH4";
        A               1.1e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-191
    {
        type            irreversibleArrheniusReaction;
        reaction        "C5H9 = aC3H5 + C2H4";
        A               2.5e+13;
        beta            0;
        Ta              15105.4;
    }
    un-named-reaction-192
    {
        type            irreversibleArrheniusReaction;
        reaction        "C5H9 = C2H3 + C3H6";
        A               2.5e+13;
        beta            0;
        Ta              15105.4;
    }
    un-named-reaction-193
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C5H10 + H = PXC5H11";
        k0
        {
            A               6.26e+32;
            beta            -6.66;
            Ta              3522.34;
        }
        kInf
        {
            A               1.33e+10;
            beta            0;
            Ta              1640.76;
        }
        F
        {
            alpha           1;
            Tsss            1000;
            Ts              1310;
            Tss             48097;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-194
    {
        type            reversibleArrheniusReaction;
        reaction        "C5H10 + H = C2H4 + nC3H7";
        A               8e+18;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-195
    {
        type            reversibleArrheniusReaction;
        reaction        "C5H10 + H = C3H6 + C2H5";
        A               1.6e+19;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-196
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + nC3H7 = PXC5H11";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-197
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C6H12 + H = PXC6H13";
        k0
        {
            A               6.26e+32;
            beta            -6.66;
            Ta              3522.34;
        }
        kInf
        {
            A               1.33e+10;
            beta            0;
            Ta              1640.76;
        }
        F
        {
            alpha           1;
            Tsss            1000;
            Ts              1310;
            Tss             48097;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-198
    {
        type            reversibleArrheniusReaction;
        reaction        "C6H12 + H = C2H4 + pC4H9";
        A               8e+18;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-199
    {
        type            reversibleArrheniusReaction;
        reaction        "C6H12 + H = C3H6 + nC3H7";
        A               1.6e+19;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-200
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + pC4H9 = PXC6H13";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-201
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C7H14 + H = PXC7H15";
        k0
        {
            A               6.26e+32;
            beta            -6.66;
            Ta              3522.34;
        }
        kInf
        {
            A               1.33e+10;
            beta            0;
            Ta              1640.76;
        }
        F
        {
            alpha           1;
            Tsss            1000;
            Ts              1310;
            Tss             48097;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-202
    {
        type            reversibleArrheniusReaction;
        reaction        "C7H14 + H = C2H4 + PXC5H11";
        A               8e+18;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-203
    {
        type            reversibleArrheniusReaction;
        reaction        "C7H14 + H = C3H6 + pC4H9";
        A               1.6e+19;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-204
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + PXC5H11 = PXC7H15";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-205
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C8H16 + H = PXC8H17";
        k0
        {
            A               6.26e+32;
            beta            -6.66;
            Ta              3522.34;
        }
        kInf
        {
            A               1.33e+10;
            beta            0;
            Ta              1640.76;
        }
        F
        {
            alpha           1;
            Tsss            1000;
            Ts              1310;
            Tss             48097;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-206
    {
        type            reversibleArrheniusReaction;
        reaction        "C8H16 + H = C2H4 + PXC6H13";
        A               8e+18;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-207
    {
        type            reversibleArrheniusReaction;
        reaction        "C8H16 + H = C3H6 + PXC5H11";
        A               1.6e+19;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-208
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + PXC6H13 = PXC8H17";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-209
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C9H18 + H = PXC9H19";
        k0
        {
            A               6.26e+32;
            beta            -6.66;
            Ta              3522.34;
        }
        kInf
        {
            A               1.33e+10;
            beta            0;
            Ta              1640.76;
        }
        F
        {
            alpha           1;
            Tsss            1000;
            Ts              1310;
            Tss             48097;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-210
    {
        type            reversibleArrheniusReaction;
        reaction        "C9H18 + H = C2H4 + PXC7H15";
        A               8e+18;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-211
    {
        type            reversibleArrheniusReaction;
        reaction        "C9H18 + H = C3H6 + PXC6H13";
        A               1.6e+19;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-212
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + PXC7H15 = PXC9H19";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-213
    {
        type            reversibleArrheniusTroeFallOffReaction;
        reaction        "C10H20 + H = PXC10H21";
        k0
        {
            A               6.26e+32;
            beta            -6.66;
            Ta              3522.34;
        }
        kInf
        {
            A               1.33e+10;
            beta            0;
            Ta              1640.76;
        }
        F
        {
            alpha           1;
            Tsss            1000;
            Ts              1310;
            Tss             48097;
        }
        thirdBodyEfficiencies
        {
            coeffs
54
(
(N2 1)
(AR 0.7)
(H 1)
(O 1)
(OH 1)
(HO2 1)
(H2 2)
(H2O 6)
(H2O2 1)
(O2 1)
(CH2 1)
(CH2* 1)
(CH3 1)
(CH4 2)
(HCO 1)
(CH2O 1)
(CH3O 1)
(CO 1.5)
(CO2 2)
(C2H2 1)
(C2H3 1)
(C2H4 1)
(C2H5 1)
(C2H6 3)
(CH2CHO 1)
(aC3H5 1)
(C3H6 1)
(nC3H7 1)
(C2H3CHO 1)
(C4H7 1)
(C4H81 1)
(pC4H9 1)
(C5H9 1)
(C5H10 1)
(PXC5H11 1)
(C6H12 1)
(PXC6H13 1)
(C7H14 1)
(PXC7H15 1)
(C8H16 1)
(PXC8H17 1)
(C9H18 1)
(PXC9H19 1)
(C10H20 1)
(PXC10H21 1)
(NC12H26 1)
(PXC12H25 1)
(SXC12H25 1)
(S3XC12H25 1)
(C12H24 1)
(C12H25O2 1)
(C12OOH 1)
(O2C12H24OOH 1)
(OC12H23OOH 1)
)
;
        }
    }
    un-named-reaction-214
    {
        type            reversibleArrheniusReaction;
        reaction        "C10H20 + H = C2H4 + PXC8H17";
        A               8e+18;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-215
    {
        type            reversibleArrheniusReaction;
        reaction        "C10H20 + H = C3H6 + PXC7H15";
        A               1.6e+19;
        beta            -2.39;
        Ta              5625.68;
    }
    un-named-reaction-216
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + PXC8H17 = PXC10H21";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-217
    {
        type            reversibleArrheniusReaction;
        reaction        "C12H24 = PXC7H15 + C5H9";
        A               3.5e+16;
        beta            0;
        Ta              35694.8;
    }
    un-named-reaction-218
    {
        type            reversibleArrheniusReaction;
        reaction        "C2H4 + PXC10H21 = PXC12H25";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-219
    {
        type            reversibleArrheniusReaction;
        reaction        "PXC12H25 = S3XC12H25";
        A               3.67e+12;
        beta            -0.6;
        Ta              7245.95;
    }
    un-named-reaction-220
    {
        type            reversibleArrheniusReaction;
        reaction        "C3H6 + PXC9H19 = SXC12H25";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-221
    {
        type            reversibleArrheniusReaction;
        reaction        "C4H81 + PXC8H17 = SXC12H25";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-222
    {
        type            reversibleArrheniusReaction;
        reaction        "C5H10 + PXC7H15 = S3XC12H25";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-223
    {
        type            reversibleArrheniusReaction;
        reaction        "C10H20 + C2H5 = S3XC12H25";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-224
    {
        type            reversibleArrheniusReaction;
        reaction        "C6H12 + PXC6H13 = S3XC12H25";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-225
    {
        type            reversibleArrheniusReaction;
        reaction        "C9H18 + nC3H7 = S3XC12H25";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-226
    {
        type            reversibleArrheniusReaction;
        reaction        "C7H14 + PXC5H11 = S3XC12H25";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-227
    {
        type            reversibleArrheniusReaction;
        reaction        "C8H16 + pC4H9 = S3XC12H25";
        A               3e+08;
        beta            0;
        Ta              3673.3;
    }
    un-named-reaction-228
    {
        type            reversibleArrheniusReaction;
        reaction        "PXC10H21 + C2H5 = NC12H26";
        A               1.88e+11;
        beta            -0.5;
        Ta              0;
    }
    un-named-reaction-229
    {
        type            reversibleArrheniusReaction;
        reaction        "PXC9H19 + nC3H7 = NC12H26";
        A               1.88e+11;
        beta            -0.5;
        Ta              0;
    }
    un-named-reaction-230
    {
        type            reversibleArrheniusReaction;
        reaction        "PXC8H17 + pC4H9 = NC12H26";
        A               1.88e+11;
        beta            -0.5;
        Ta              0;
    }
    un-named-reaction-231
    {
        type            reversibleArrheniusReaction;
        reaction        "PXC7H15 + PXC5H11 = NC12H26";
        A               1.88e+11;
        beta            -0.5;
        Ta              0;
    }
    un-named-reaction-232
    {
        type            reversibleArrheniusReaction;
        reaction        "PXC6H13 + PXC6H13 = NC12H26";
        A               1.88e+11;
        beta            -0.5;
        Ta              0;
    }
    un-named-reaction-233
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + H = PXC12H25 + H2";
        A               1300;
        beta            2.54;
        Ta              3399.56;
    }
    un-named-reaction-234
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + H = SXC12H25 + H2";
        A               2600;
        beta            2.4;
        Ta              2249.77;
    }
    un-named-reaction-235
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + H = S3XC12H25 + H2";
        A               3900;
        beta            2.4;
        Ta              2249.77;
    }
    un-named-reaction-236
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + O = PXC12H25 + OH";
        A               190;
        beta            2.68;
        Ta              1869.86;
    }
    un-named-reaction-237
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + O = SXC12H25 + OH";
        A               95.2;
        beta            2.71;
        Ta              1059.72;
    }
    un-named-reaction-238
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + O = S3XC12H25 + OH";
        A               142.8;
        beta            2.71;
        Ta              1059.72;
    }
    un-named-reaction-239
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + OH = PXC12H25 + H2O";
        A               3.4;
        beta            2.66;
        Ta              265.182;
    }
    un-named-reaction-240
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + OH = SXC12H25 + H2O";
        A               74;
        beta            2.39;
        Ta              197.754;
    }
    un-named-reaction-241
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + OH = S3XC12H25 + H2O";
        A               101;
        beta            2.39;
        Ta              197.754;
    }
    un-named-reaction-242
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + O2 = PXC12H25 + HO2";
        A               4e+10;
        beta            0;
        Ta              25627.5;
    }
    un-named-reaction-243
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + O2 = SXC12H25 + HO2";
        A               8e+10;
        beta            0;
        Ta              23946.9;
    }
    un-named-reaction-244
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + O2 = S3XC12H25 + HO2";
        A               1.2e+11;
        beta            0;
        Ta              23946.9;
    }
    un-named-reaction-245
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + HO2 = PXC12H25 + H2O2";
        A               67.6;
        beta            2.55;
        Ta              8297.62;
    }
    un-named-reaction-246
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + HO2 = SXC12H25 + H2O2";
        A               89;
        beta            2.6;
        Ta              6999.39;
    }
    un-named-reaction-247
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + HO2 = S3XC12H25 + H2O2";
        A               88.5;
        beta            2.6;
        Ta              6999.39;
    }
    un-named-reaction-248
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + CH3 = PXC12H25 + CH4";
        A               0.00181;
        beta            3.65;
        Ta              3599.33;
    }
    un-named-reaction-249
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + CH3 = SXC12H25 + CH4";
        A               0.006;
        beta            3.46;
        Ta              2757.49;
    }
    un-named-reaction-250
    {
        type            reversibleArrheniusReaction;
        reaction        "NC12H26 + CH3 = S3XC12H25 + CH4";
        A               0.009;
        beta            3.46;
        Ta              2757.49;
    }
    un-named-reaction-251
    {
        type            irreversibleArrheniusReaction;
        reaction        "PXC12H25 + O2 = C12H25O2";
        A               5e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-252
    {
        type            irreversibleArrheniusReaction;
        reaction        "C12H25O2 = PXC12H25 + O2";
        A               2.75e+13;
        beta            0;
        Ta              13787.4;
    }
    un-named-reaction-253
    {
        type            irreversibleArrheniusReaction;
        reaction        "SXC12H25 + O2 = C12H25O2";
        A               5e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-254
    {
        type            irreversibleArrheniusReaction;
        reaction        "C12H25O2 = SXC12H25 + O2";
        A               2.75e+13;
        beta            0;
        Ta              13787.4;
    }
    un-named-reaction-255
    {
        type            irreversibleArrheniusReaction;
        reaction        "S3XC12H25 + O2 = C12H25O2";
        A               5e+10;
        beta            0;
        Ta              0;
    }
    un-named-reaction-256
    {
        type            irreversibleArrheniusReaction;
        reaction        "C12H25O2 = S3XC12H25 + O2";
        A               2.75e+13;
        beta            0;
        Ta              13787.4;
    }
    un-named-reaction-257
    {
        type            irreversibleArrheniusReaction;
        reaction        "C12H25O2 = C12OOH";
        A               1.51e+12;
        beta            0;
        Ta              9560.63;
    }
    un-named-reaction-258
    {
        type            irreversibleArrheniusReaction;
        reaction        "C12OOH = C12H25O2";
        A               1e+11;
        beta            0;
        Ta              5786.7;
    }
    un-named-reaction-259
    {
        type            irreversibleArrheniusReaction;
        reaction        "PXC12H25 + O2 = C12H24 + HO2";
        A               3.5e+08;
        beta            0;
        Ta              3019.15;
    }
    un-named-reaction-260
    {
        type            irreversibleArrheniusReaction;
        reaction        "C12H24 + HO2 = PXC12H25 + O2";
        A               3.16e+08;
        beta            0;
        Ta              9812.23;
    }
    un-named-reaction-261
    {
        type            irreversibleArrheniusReaction;
        reaction        "SXC12H25 + O2 = C12H24 + HO2";
        A               3.5e+08;
        beta            0;
        Ta              3019.15;
    }
    un-named-reaction-262
    {
        type            irreversibleArrheniusReaction;
        reaction        "C12H24 + HO2 = SXC12H25 + O2";
        A               3.16e+08;
        beta            0;
        Ta              9812.23;
    }
    un-named-reaction-263
    {
        type            irreversibleArrheniusReaction;
        reaction        "S3XC12H25 + O2 = C12H24 + HO2";
        A               3.5e+08;
        beta            0;
        Ta              3019.15;
    }
    un-named-reaction-264
    {
        type            irreversibleArrheniusReaction;
        reaction        "C12H24 + HO2 = S3XC12H25 + O2";
        A               3.16e+08;
        beta            0;
        Ta              9812.23;
    }
    un-named-reaction-265
    {
        type            irreversibleArrheniusReaction;
        reaction        "C12OOH + O2 = O2C12H24OOH";
        A               4.6e+07;
        beta            0;
        Ta              0;
    }
    un-named-reaction-266
    {
        type            irreversibleArrheniusReaction;
        reaction        "O2C12H24OOH = C12OOH + O2";
        A               2.51e+13;
        beta            0;
        Ta              13787.4;
    }
    un-named-reaction-267
    {
        type            reversibleArrheniusReaction;
        reaction        "O2C12H24OOH = OC12H23OOH + OH";
        A               8.9e+10;
        beta            0;
        Ta              8554.25;
    }
    un-named-reaction-268
    {
        type            irreversibleArrheniusReaction;
        reaction        "OC12H23OOH = 3C2H4 + C2H5 + 2CH2CHO + OH";
        A               1.8e+15;
        beta            0;
        Ta              21166.7;
    }
}
Tlow            200;
Thigh           5000;)"