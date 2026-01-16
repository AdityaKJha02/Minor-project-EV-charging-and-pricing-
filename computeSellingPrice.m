function selling_price = computeSellingPrice(rtp_price, markup_factor)
% computeSellingPrice
% This function calculates the selling price of electricity
% based on a 24-hour RTP signal and a markup factor.
%
% Inputs:
%   rtp_price     - 1x24 or 24x1 array of RTP values (Rs/kWh)
%   markup_factor - scalar markup factor (e.g., 0.2 for 20%)
%
% Output:
%   selling_price - array of selling prices for 24 hours (Rs/kWh)

    % Ensure RTP is a column vector (optional, for consistency)
    rtp_price = rtp_price(:);

    % Calculate selling price
    selling_price = (1 + markup_factor) .* rtp_price;

end

