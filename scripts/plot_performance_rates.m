% plot_performance_rates(recall_rates, precision_rates, color='r')
%
% This function plots the precision and recall rates of a single query
% for a varying number of candidates.
%
% Input:
%	rankedFile: path to the CSV ranked file holding precision and recall rates
%	color: format of the ploted curves
%

function plot_performance_rates(rankedFile, color)

	data = csvread(rankedFile);

	precision_rates = data(1, :);
	recall_rates = data(2, :);

	if nargin < 1
		help plot_performance_rates;
	end

	if nargin < 2
	    color = 'r';
	end

	line_format = strcat(color, '-d');

	% Plot avg precision and recall
	subplot(3,1,1), hold on, plot(1:length(recall_rates), recall_rates, line_format, 'MarkerFaceColor', color, 'MarkerSize', 3), grid on, hold off, xlabel('Number of candidates'), ylabel('Recall');
	subplot(3,1,2), hold on, plot(1:length(precision_rates), precision_rates, line_format, 'MarkerFaceColor', color, 'MarkerSize', 3), grid on, hold off, xlabel('Number of candidates'), ylabel('1-Precision');
	subplot(3,1,3), hold on, plot(recall_rates, precision_rates, line_format, 'MarkerFaceColor', color, 'MarkerSize', 3), hold off, grid on, ylabel('Precision'), xlabel('Recall');

end
