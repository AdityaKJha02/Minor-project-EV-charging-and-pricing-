function e_bar = getAverageSessionEnergy(EVtimeline_file, sheetName)
% Returns average energy (kWh) per EV session using EV timeline sheet.
% Expects the EV timeline sheet where column L contains energy_kWh.
%
% If no sessions found, returns 0.

T = readtable(EVtimeline_file,'Sheet',sheetName);

% Column L is column 12 in your screenshot; adapt if different
if size(T,2) < 12
    warning('Timeline sheet %s has fewer than 12 columns; returning e_bar=0', sheetName);
    e_bar = 0;
    return;
end

energyCol = T{:,12};         % numeric vector (energy_kWh)
% remove NaN / non-positive entries
energyCol = energyCol(~isnan(energyCol) & energyCol > 0);

if isempty(energyCol)
    e_bar = 0;
else
    e_bar = sum(energyCol) / numel(energyCol);
end
end
