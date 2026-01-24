function nEV = getEVcountAtTime(EV_file, sheetNames_EV, fcsID, time_min)
% =====================================================
% Returns number of EVs charging at a given FCS and time
%
% INPUTS:
%   EV_file        : Excel file name
%   sheetNames_EV  : cell array of sheet names
%   fcsID          : FCS number (1 to nFCS)
%   time_min       : time in minutes (0–1430, multiple of 10)
%
% OUTPUT:
%   nEV            : number of EVs charging
% =====================================================

% ---- Read sheet for selected FCS ----
sheetName = sheetNames_EV{fcsID};
T = readtable(EV_file, 'Sheet', sheetName);

% ---- Column D contains EV IDs ----
EVcol = T{:,4};   % column D

% ---- Find time index ----
% Time column is assumed in minutes (column C)
timeCol = T{:,3};

idx = find(timeCol == time_min, 1);

if isempty(idx)
    nEV = 0;
    return;
end

cellVal = EVcol{idx};

% ---- Count EVs ----
if isempty(cellVal) || (isstring(cellVal) && strlength(cellVal)==0)
    nEV = 0;
else
    parts = split(string(cellVal), ",");
    nEV = numel(parts);
end

end
