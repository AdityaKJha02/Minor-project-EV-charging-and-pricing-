function delta_t = charging_time(SOC1, SOC2)
    t = [0 10 15 22 34 60];
    soc = [20 50 75 85 95 100];

    t1 = interp1(soc, t, SOC1, 'linear');
    t2 = interp1(soc, t, SOC2, 'linear');

    delta_t = t2 - t1;
end
