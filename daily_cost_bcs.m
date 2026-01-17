function Cd = daily_cost_bcs(Ehev, rho_h, r, n, params)
% =========================================================
% DAILY COST OF BASE CHARGING STATION (BCS)
% =========================================================

    gamma = 0.05;   % O&M coefficient

    % Capital Recovery Factor
    CRF = (r * (1 + r)^n) / (((1 + r)^n) - 1);

    % Fixed daily cost
    C_FIX = (1/30) * CRF * ( ...
        params.A_fc   * params.U_L   + ...
        params.N_CO   * params.U_BDG + ...
        params.beta_SS * params.U_SS );

    % O&M cost
    C_OandM = gamma * C_FIX;

    % Energy purchase cost
    C_energy = sum(Ehev .* rho_h);
    disp(C_energy)

    % Total daily cost
    Cd = C_FIX + C_OandM + C_energy;

end
