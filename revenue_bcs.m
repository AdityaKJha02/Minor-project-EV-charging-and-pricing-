function RD = revenue_bcs(Ehev, rho_h, markup_factor)
% =========================================================
% daily_revenue_dp
% =========================================================
% Calculates DAILY REVENUE for ONE EVFCS
%
% INPUTS:
% Ehev          : Energy sold vector (kWh) [Nx1]
% rho_h         : Grid purchase price (Rs/kWh) [Nx1]
% markup_factor : Markup factor (alpha)
%
% OUTPUT:
% RD            : Daily revenue (Rs/day)
%
% Eq:
% rho_SELL = (1 + alpha) * rho_PUR
% R^D = sum(Ehev .* rho_SELL)
% =========================================================

    % Ensure column vectors
    Ehev  = Ehev(:);
    rho_h = rho_h(:);

    % Selling price
    selling_price = computeSellingPrice(rho_h, markup_factor);

    % Revenue
    RD = sum(Ehev .* selling_price);

end

