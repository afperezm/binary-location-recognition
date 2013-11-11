% plot_performance_rates(recall_rates, precision_rates, line_format='r-o')
%
% This function plots the average precision and recall rates computed previously
% starting from the matrix of right landmark ids occurrences.
%
% Input:
%	recall_rates: average recall values for a varying number of candidates
%	precision_rates: average precision values for a varying number of candidates
%	line_format: format of the ploted curves
%

function plot_performance_rates(recall_rates, precision_rates, color)

	if nargin < 3
	    color = 'r';
	end

	line_format = strcat(color, '-d');

	% Plot avg precision and recall
	subplot(3,1,1), hold on, plot(1:length(recall_rates), recall_rates, line_format, 'MarkerFaceColor', color, 'MarkerSize', 3), hold off, xlabel('Number of candidates'), ylabel('Recall');
	subplot(3,1,2), hold on, plot(1:length(precision_rates), precision_rates, line_format, 'MarkerFaceColor', color, 'MarkerSize', 3), hold off, xlabel('Number of candidates'), ylabel('1-Precision');
	subplot(3,1,3), hold on, plot(recall_rates, precision_rates, line_format, 'MarkerFaceColor', color, 'MarkerSize', 3), hold off, ylabel('Precision'), xlabel('Recall');

end
