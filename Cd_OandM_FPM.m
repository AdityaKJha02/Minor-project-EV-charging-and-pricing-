function C_OandM = Cd_OandM_FPM(C_FIX)
% =====================================================
% O&M COST COMPUTATION FOR FPM
% Implements: C_O&M = φ × C_FIX   (Eq. 31)
%
% INPUT:
%   C_FIX   : [1×nFCS] fixed daily cost from Cd_fixcost_FPM
%
% OUTPUT:
%   C_OandM : [1×nFCS] O&M daily cost (Rs/day)
% =====================================================

phi = 0.05;   % O&M coefficient (paper)

C_OandM = phi * C_FIX;

end
