function edgeOccRounded = totalPreservingHourlyEVs(edgeOcc)
% edgeOcc: [numEdges x 24] fractional occupancy
% edgeOccRounded: integer occupancy with row-wise total preserved

[numEdges, numHours] = size(edgeOcc);
edgeOccRounded = zeros(numEdges, numHours);

for e = 1:numEdges

    row = edgeOcc(e,:);

    % Original total (robust to floating error)
    targetTotal = round(sum(row));

    % Step 1: floor
    base = floor(row);

    % Step 2: remaining EVs to assign
    remainder = row - base;
    missing = targetTotal - sum(base);

    % Step 3: distribute remaining EVs
    [~, idx] = sort(remainder, 'descend');

    add = zeros(1, numHours);
    if missing > 0
        add(idx(1:missing)) = 1;
    end

    % Final rounded row
    edgeOccRounded(e,:) = base + add;
end

end
