% plot_overall_performance(ranked_files_folder, num_queries, color='r')
%
% This function plots the average precision and recall rates of a set of queries
% for a varying number of candidates.
%
% Input:
%	ranked_files_folder: path to the folder of ranked files
%	num_queries: number of queries over which to take the average
%	color: color of the plotted line
%

function plot_overall_performance(ranked_files_folder, num_queries, color)

	if nargin < 2
		help plot_overall_performance;
	end

	if nargin < 3
	    color = 'r';
	end

	avg_recall_rates = 0;
	avg_precision_rates = 0;

	for queryId=0:1:num_queries-1
		rankedFilename = strcat(ranked_files_folder, "/", "query_", num2str(queryId), ".csv");

		data = csvread(rankedFilename);

		avg_recall_rates += data(2,:);
		avg_precision_rates += data(1,:);
	end

	avg_recall_rates/=num_queries;
	avg_precision_rates/=num_queries;

	plot(avg_recall_rates, avg_precision_rates, 'Color', color, 'LineStyle','-'), grid on, ylabel('Precision'), xlabel('Recall');

end
