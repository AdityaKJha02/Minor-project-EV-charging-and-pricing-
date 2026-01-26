function P_144 = buildCentralPeriodicLoadCurve( ...
        EV_file, sheetName, powerCol, nSlots)
% =========================================================
% Build 144-slot periodic EV load curve from longer data
% - Selects CENTER nSlots
% - Clips negative power to zero
% - Enforces P(1) = P(end)
% =========================================================
%
% INPUTS:
%   EV_file   : Excel file
%   sheetName: Sheet name
%   powerCol : Column index for EV power (kW)
%   nSlots   : Desired slots (144)
%
% OUTPUT:
%   P_144    : [nSlots × 1] periodic EV power (kW)
% =========================================================

    % ---------------------------
    % Read power column
    % ---------------------------
    M = readmatrix(EV_file, 'Sheet', sheetName);
    P = M(:, powerCol);

    % Remove NaNs
    P = P(~isnan(P));

    N = numel(P);
    if N < nSlots
        error('Sheet %s has only %d slots (< %d)', sheetName, N, nSlots);
    end

    % ---------------------------
    % Select CENTRAL nSlots
    % ---------------------------
    midIdx = floor(N/2);
    startIdx = midIdx - floor(nSlots/2) + 1;
    endIdx   = startIdx + nSlots - 1;

    if startIdx < 1
        startIdx = 1;
        endIdx   = nSlots;
    end
    if endIdx > N
        endIdx   = N;
        startIdx = N - nSlots + 1;
    end

    P_144 = P(startIdx:endIdx);

    % ---------------------------
    % Enforce non-negativity
    % ---------------------------
    P_144(P_144 < 0) = 0;

    % ---------------------------
    % Enforce periodicity
    % ---------------------------
    P_144(end) = P_144(1);

    % Ensure column vector
    P_144 = P_144(:);
end
