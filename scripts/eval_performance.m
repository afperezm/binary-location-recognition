% Evaluate performance of the recognition pipeline

function eval_performance(data_file)

data = csvread(data_file);

precision = data(1,:);
recall = data(2,:);
plot_performance_rates(recall, precision);
