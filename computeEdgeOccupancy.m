function edgeOcc = computeEdgeOccupancy( ...
    taskPaths, vehFlowRatio, maxEVs, ...
    edgeMap, distMatrix, kmPerUnit, speed_kmh, numHours)

numEdges = size(edgeMap,1);
edgeOcc  = zeros(numEdges, numHours);

for hStart = 1:numHours

    % EVs generated per task at hour hStart
    EVsPerTask = vehFlowRatio(hStart) * maxEVs;
    if EVsPerTask == 0
        continue;
    end

    % Continuous start time (hours)
    t0 = hStart - 1;

    for k = 1:numel(taskPaths)
        path = taskPaths{k};
        t = t0;

        for n = 1:length(path)-1
            from = path(n);
            to   = path(n+1);

            % locate edge
            idx = find(edgeMap(:,2)==from & edgeMap(:,3)==to);
            if isempty(idx)
                break;
            end

            % travel time on this edge (hours)
            travel_hr = (distMatrix(from,to) * kmPerUnit) / speed_kmh;

            t_in  = t;
            t_out = t + travel_hr;

            % hours overlapped by this traversal
            hFirst = max(1, floor(t_in) + 1);
            hLast  = min(numHours, ceil(t_out));

            for h = hFirst:hLast
                % overlap condition
                if t_in < h && t_out > (h-1)
                    edgeOcc(idx,h) = edgeOcc(idx,h) + EVsPerTask;
                end
            end

            t = t_out;  % move to next edge
        end
    end
end

end
