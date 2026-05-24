filename = 'EVFCS_10min_Arrival_Power_tou (2).xlsx';

sheetNames_EV = {    % dpm
    'EVFCS_1_6_2'
    'EVFCS_2_24_21'
    'EVFCS_3_15_22'
    'EVFCS_4_12_13'
    'EVFCS_5_7_18'
    'EVFCS_6_10_15'
};

for s = 1:length(sheetNames_EV)

    sheet = sheetNames_EV{s};

    % Read sheet
    data = readcell(filename,'Sheet',sheet);

    colG = 7; % Column G

    firstRow = [];

    % Find first non-zero value in column G
    for r = 2:size(data,1)
        val = data{r,colG};

        if isnumeric(val) && ~isnan(val) && val ~= 0
            firstRow = r;
            break
        end
    end

    % Move first valid row to row 2
    if ~isempty(firstRow) && firstRow > 2
        data = [data(1,:); data(firstRow:end,:)];
    end
    % Number of data rows (excluding header)
n = size(data,1) - 1;

% Generate time sequence
time_min = (0:10:(n-1)*10)';

hour = mod(floor(time_min/60),24);
minute = mod(time_min,60);

% Rewrite first three columns
data(2:end,1) = num2cell(hour);
data(2:end,2) = num2cell(minute);
data(2:end,3) = num2cell(time_min);

    % --- Fix for writecell ---
    for i = 1:numel(data)
        if ismissing(data{i})
            data{i} = [];
        end
    end
    % -------------------------

    % Write back to Excel
    writecell(data, filename,'Sheet', sheet);

end

disp('All sheets processed successfully.');
