function beta_SPV = compute_beta_SPV(N_co)
% =========================================================
% Computes rooftop SPV capacity beta_SPV using Eq. (37)
%
% beta_SPV = A_fc * p * cos(phi)
%
% INPUT:
%   N_co : number of chargers at the FCS
%
% OUTPUT:
%   beta_SPV : rooftop SPV capacity (W)
% =========================================================

%% ---------------------------
% GIVEN PARAMETERS (FROM PAPER)
% ---------------------------
phi_deg  = 40;      % tilt angle (degrees)
capacity = .550;     % panel capacity (W)
m        = 0.75;    % usable rooftop fraction

% Panel dimensions (cm)
length_cm = 227.9;
width_cm  = 113.4;

%% ---------------------------
% PANEL AREA (m^2)
% ---------------------------
length_m = length_cm / 100;
width_m  = width_cm  / 100;
panel_area = length_m * width_m;   % m^2

%% ---------------------------
% POWER DENSITY p (W/m^2)
% ---------------------------
p = capacity / panel_area;

%% ---------------------------
% ROOFTOP AREA FROM YOUR FUNCTION
% ---------------------------
% Area_calc already includes charger/layout logic
A_fcn = Area_calc(N_co);    % m^2

% Effective rooftop area
A_fc = m * A_fcn;

%% ---------------------------
% SPV CAPACITY (Eq. 37)
% ---------------------------
beta_SPV = A_fc * p * cosd(phi_deg);

end
