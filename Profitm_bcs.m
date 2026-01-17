function PM = Profitm_bcs(RD, Cd)
% =========================================================
% daily_profit_margin
% =========================================================
% Calculates DAILY PROFIT MARGIN (%)
%
% INPUTS:
% RD : Daily revenue (Rs/day)
% Cd : Daily cost (Rs/day)
%
% OUTPUT:
% PM : Profit margin (%) 
%
% Formula:
% PM = 100 * (RD - Cd) / RD
% =========================================================

    if RD == 0
        PM = 0;   % Avoid division by zero
    else
        PM = 100 * (RD - Cd) / RD;
    end

end
