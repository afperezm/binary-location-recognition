%

addpath('/home/andresf/workspace-cpp/VLRPipeline/scripts');

% Setup figure

h = figure(1);
H = 3;
W = 4;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

% 1. Find best vocabulary parameters - Hierarchical K-Majority - Vocabulary Size

%words = [32768 100000 248832 262144 537824 1000000 1048576 2097152 2985984 7529536 10000000 16777216 35831808 105413504 268435456];
%mAP = [0.134532 0.143451 0.152825 0.156246 0.146797 0.162759 0.162513 0.161149 0.156529 0.167971 0.169227 0.16367 0.167933 0.162788 0.163898];
%plot(words, mAP, "Color", "b", "LineStyle", "-", "LineWidth", 2, "Marker", "o");

hold on;
words = [32768 100000 248832 537824 1048576]
mAP = [0.134532 0.143451 0.152825 0.146797 0.162513]
semilogx(words, mAP, "Color", "blue", "LineStyle", "-", "LineWidth", 2, "Marker", "o");
words = [262144 1000000 2985984 7529536 16777216]
mAP = [0.156246 0.162759 0.156529 0.167971 0.16367]
semilogx(words, mAP, "Color", "red", "LineStyle", "-.", "LineWidth", 2, "Marker", "o");
words = [2097152 10000000 35831808 105413504 268435456]
mAP = [0.161149 0.169227 0.167933 0.162788 0.163898]
semilogx(words, mAP, "Color", "black", "LineStyle", "--", "LineWidth", 2, "Marker", "o");
hold off;

grid on;

xlabel("Number of words");
ylabel("Mean Average Precision");

legend("5", "6", "7", "location", "southeast");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 12);

print(h, "-dpng", "-color", "map_test_hkmaj_nwords.png");

% 1. Find best vocabulary parameters - Hierarchical K-Majority - training cycles

maxIterations = [1 5 10 15 20 25 30 35 40 45 50];
mAP = [0.128658 0.164739 0.162759 0.160819 0.164337 0.156415 0.172099 0.160351 0.161977 0.164023 0.16178];

plot(maxIterations, mAP, "Color", "b", "LineStyle", "-", "LineWidth", 2, "Marker", "o");

grid on;

xlabel("Maximum number of training cycles");
ylabel("Mean Average Precision");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 12);

print(h, "-dpng", "-color", "map_test_hkmaj_max_iterations.png");

% 1. Find best vocabulary parameters - Hierarchical K-Majority - training data

dataSize = [5 25 50 75 100];
mAP = [0.160105 0.157795 0.165851 0.169869 0.159483];

plot(dataSize, mAP, "Color", "b", "LineStyle", "-", "LineWidth", 2, "Marker", "o");

grid on;

xlabel("Amount of training data (%)");
ylabel("Mean Average Precision");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 12);

print(h, "-dpng", "-color", "map_test_hkmaj_data_size.png");

% 1. Find best vocabulary parameters - Hierarchical K-Majority - Shape

h = figure(1);
H = 3;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

hold on;
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_hkmaj_shape/deep", 55, "r", "-");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_hkmaj_shape/super", 55, "b", "-.");
hold off;

legend("Depth 6, Branching 16", "Depth 8, Branching 8")

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 12);

print(h, "-dpng", "-color", "map_test_hkmaj_shape.png");

% 1. Find best vocabulary parameters - Hierarchical K-Majority - Seeding

h = figure(1);
H = 3;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

hold on;
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_hkmaj_seeding/gonzalez", 55, "b", "-.");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_hkmaj_seeding/random", 55, "r", "-");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_hkmaj_seeding/kmeanspp", 55, "g", "--");
hold off;

legend("Gonzalez algorithm", "Random", "K-Means++");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 12);

print(h, "-dpng", "-color", "map_test_hkmaj_seeding.png");

% 1. Find best vocabulary parameters - Hierarchical K-Majority - Best scheme

h = figure(1);
H = 6;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

hold on;
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_ref", 55, "blue", "-");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_base", 55, "green", "--");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_hkmaj_cycles/", 55, "black", "-.");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_hkmaj_best", 55, "red", ":");
hold off;

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 24);

L = legend("Reference", "Baseline", "Best max iterations", "Combined scheme");

fontName = findall(L, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(L, "-property", "FontSize");
set(fontSize, "FontSize", 16);

print(h, "-dpng", "-color", "map_test_hkmaj_best.png");

% Bad performing queries

ap_ref = [0.576420 0.581057 0.525852 0.654214 0.614274 0.541385 0.411016 0.349108 0.401176 0.464551 0.521218 0.208824 0.058810 0.304801 0.263596 0.632600 0.721490 0.663759 0.689228 0.709917 0.720773 0.725170 0.666597 0.741831 0.155507 0.120628 0.008795 0.118769 0.053330 0.245125 0.667434 0.654200 0.610420 0.706713 0.660187 0.285714 0.285714 0.226190 0.180952 0.285714 0.079149 0.303047 0.208284 0.234091 0.190268 0.013889 0.166667 0.166667 0.166667 0.166667 0.819549 0.876566 0.809974 0.832965 0.792720];

ap_base= [0.272503 0.441234 0.418322 0.603427 0.375244 0.570303 0.437526 0.373236 0.325691 0.400996 0.191543 0.048476 0.102696 0.014256 0.282878 0.468777 0.674948 0.553631 0.661521 0.651366 0.684025 0.700537 0.613464 0.684820 0.087345 0.119905 0.010489 0.113623 0.009027 0.013342 0.560512 0.481091 0.504078 0.625234 0.389160 0.285714 0.285714 0.285714 0.285714 0.285714 0.083440 0.316270 0.142207 0.217957 0.102988 0.005208 0.166667 0.008333 0.004902 0.027778 0.615858 0.630165 0.613729 0.588557 0.619838];

ap_hkmaj_cycles = [0.280925 0.216404 0.381714 0.366418 0.357986 0.490464 0.243106 0.054946 0.145072 0.137773 0.013672 0.011214 0.011389 0.006399 0.024211 0.268166 0.106964 0.114540 0.460897 0.474571 0.287465 0.364087 0.228979 0.246469 0.063106 0.004139 0.001957 0.003003 0.003252 0.002202 0.242941 0.177792 0.231885 0.224176 0.114980 0.001348 0.004628 0.004031 0.000786 0.143345 0.042535 0.051267 0.061698 0.062418 0.068352 0.008333 0.001082 0.000538 0.001082 0.000980 0.527914 0.543159 0.589438 0.524185 0.465047];

ap_hkmaj_best = [0.217803 0.210223 0.345361 0.308719 0.318520 0.390768 0.245166 0.102376 0.159427 0.047330 0.096917 0.006972 0.019306 0.018172 0.005543 0.250771 0.120548 0.157713 0.340817 0.383005 0.275513 0.410077 0.393128 0.286001 0.063928 0.002578 0.002037 0.002281 0.003652 0.002512 0.227547 0.226558 0.165735 0.202011 0.178019 0.001420 0.004900 0.003100 0.000684 0.002087 0.060028 0.055428 0.062666 0.058954 0.064379 0.000877 0.002315 0.000288 0.000149 0.000656 0.552927 0.542187 0.624181 0.519685 0.485213];

ap_hkmaj_feat_scale = [0.221920 0.194575 0.306248 0.393284 0.388991 0.461881 0.302901 0.255526 0.237146 0.311526 0.026662 0.009726 0.020212 0.019035 0.009768 0.153375 0.118441 0.144127 0.660738 0.667393 0.342999 0.335796 0.325907 0.224357 0.056680 0.010031 0.013849 0.006751 0.116089 0.005541 0.346166 0.381440 0.186289 0.343832 0.170292 0.002191 0.012347 0.002202 0.001307 0.004879 0.059879 0.056793 0.058385 0.108198 0.050482 0.000683 0.000672 0.000476 0.001096 0.001462 0.535628 0.575061 0.622184 0.623379 0.599638];

h = figure(1);
H = 4;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

plot(ap_ref, "Color", "blue", "Marker", "*", "LineStyle", "--", "MarkerSize", 3), hold on;
plot(ap_base, "Color", "green", "Marker", "o", "LineStyle", "--", "MarkerSize", 3);
plot(ap_hkmaj_cycles, "Color", "black", "Marker", "+", "LineStyle", "--", "MarkerSize", 3);
plot(ap_hkmaj_best, "Color", "red", "Marker", "d", "LineStyle", "--", "MarkerSize", 3);
plot(ap_hkmaj_feat_scale, "Color", "magenta", "Marker", "^", "LineStyle", "--", "MarkerSize", 3), hold off;

legend("Reference", "Baseline", "DoG, BRIEF, Depth 6, Branching 10, Max cycles 10, Seeding Random", "DoG, BRIEF, Depth 7, Branching 10, Max cycles 30, Seeding K-Means++", "location", "northwest")

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 6);

% 2. Find best database parameters

h = figure(1);
H = 5;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

hold on;
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_inv_idx_0", 55, "blue");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_inv_idx_1", 55, "red");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_inv_idx_2", 55, "cyan", "-");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_inv_idx_3", 55, "black", ":");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_inv_idx_4", 55, "green");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_inv_idx_5", 55, "magenta");
hold off;

legend("TFIDF, L1, L1", "TFIDF, L2, COS", "BIN, L1, L1", "BIN, L2, L2", "None, L1, L1", "None, L2, L2");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 10);

print(h, "-dpng", "-color", "map_test_inv_idx.png");

% 3. Geometric verification

h = figure(1);
H = 5;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

hold on;
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_ref", 55, "blue", ":");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_gv_ref_2000_0.7_8_10", 55, "blue", "-");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_hkmaj_cycles", 55, "red", ":");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_gv_hkmaj_best_cycles_700_20_4_10", 55, "red", "-");
hold off;
grid on;

legend("Reference, pre-gv", "Reference, post-gv", "Best HKMaj, pre-gv", "Best HKMaj, post-gv");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 16);

print(h, "-dpng", "-color", "map_test_gv.png");

% 3. Geometric verification - top key points
h = figure(1);
H = 3;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

topKpts_ref = [500 700 900 1100 2000];
mAP_ref = [0.401922 0.414639 0.425534 0.431405 0.447327];
topKpts_hkmaj = [500 700 900];
mAP_hkmaj = [0.177048 0.180195 0.18083];

hold on;
plot(topKpts_ref, mAP_ref, "Color", "red", "Marker", "o", "LineStyle", "--", "LineWidth", 4);
plot(topKpts_hkmaj, mAP_hkmaj, "Color", "blue", "Marker", "d", "LineStyle", "--", "LineWidth", 4);
hold off;
grid on;

xlabel("Number of top keypoints");
ylabel("Mean Average Precision");

xlim([500 2000]);
legend("Reference", "Best HKMaj", "location", "east");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 10);

print(h, "-dpng", "-color", "map_test_gv_topKpts.png");

% 3. Geometric verification - ratio threshold

h = figure(1);
H = 3;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

% 500, 8, 10
ratioThr_top500 = [0.6 0.7 0.8];
mAP_top500 = [0.433052 0.42591 0.401922];
% 700, 8, 10
ratioThr_top700 = [0.6 0.7 0.8];
mAP_top700 = [0.434735 0.431581 0.414639];
% 1100, 8, 10
ratioThr_top1100 = [0.6 0.7 0.8];
mAP_top1100 = [0.438 0.440546 0.431405];
% 2000, 8, 10
ratioThr_top2000 = [0.6 0.7 0.8];
mAP_top2000 = [0.442731 0.45186 0.447327];

hold on;
plot(ratioThr_top500, mAP_top500, "Color", "red", "Marker", "o", "LineStyle", "-", "LineWidth", 4);
plot(ratioThr_top700, mAP_top700, "Color", "green", "Marker", "d", "LineStyle", "-", "LineWidth", 4);
plot(ratioThr_top1100, mAP_top1100, "Color", "blue", "Marker", "*", "LineStyle", "-", "LineWidth", 4);
plot(ratioThr_top2000, mAP_top2000, "Color", "black", "Marker", "^", "LineStyle", "-", "LineWidth", 4);
hold off;
grid on;

xlabel("SIFT ratio threshold");
ylabel("Mean Average Precision");

xlim([0.6 0.8]);
legend("500", "700", "1100", "2000", "location", "southwest");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 10);

print(h, "-dpng", "-color", "map_test_gv_ratioThr.png");

% 3. Geometric verification - distance threshold

h = figure(1);
H = 3;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

% 500, 8, 10
distThr_top500 = [5 20 90];
mAP_top500 = [0.172099 0.177048 0.163793];
% 700, 8, 10
distThr_top700 = [10 20 30];
mAP_top700 = [0.173747 0.180195 0.178148];
% 900, 8, 10
distThr_top900 = [10 20 30];
mAP_top900 = [0.174527 0.18083 0.17878];

hold on;
plot(distThr_top500, mAP_top500, "Color", "red", "Marker", "o", "LineStyle", "-", "LineWidth", 4);
plot(distThr_top700, mAP_top700, "Color", "green", "Marker", "d", "LineStyle", "-", "LineWidth", 4);
plot(distThr_top900, mAP_top900, "Color", "blue", "Marker", "*", "LineStyle", "-", "LineWidth", 4);
hold off;
grid on;

xlabel("Distance threshold");
ylabel("Mean Average Precision");

xlim([5 90]);
legend("500", "700", "900", "location", "northeast");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 10);

print(h, "-dpng", "-color", "map_test_gv_distThr.png");

% 3. Geometric verification - minimum matches

h = figure(1);
H = 3;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

minMatches_ref = [4 8 16 50];
mAP_ref = [0.437743 0.434735 0.433585 0.433459];
minMatches_hkmaj = [4 8 20];
mAP_hkmaj = [0.181457 0.180195 0.176547];

hold on;
plot(minMatches_ref, mAP_ref, "Color", "red", "Marker", "o", "LineStyle", "--", "LineWidth", 4);
plot(minMatches_hkmaj, mAP_hkmaj, "Color", "blue", "Marker", "d", "LineStyle", "--", "LineWidth", 4);
hold off;
grid on;

xlabel("Minimum number of matches");
ylabel("Mean Average Precision");

xlim([4 50]);
legend("Reference", "Best HKMaj", "location", "east");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 10);

print(h, "-dpng", "-color", "map_test_gv_minMatches.png");

% 3. Geometric verification - RANSAC threshold

h = figure(1);
H = 3;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

ransacThr_ref = [5 10 15 30];
mAP_ref = [0.434735 0.434735 0.434735 0.434735];
ransacThr_hkmaj = [3 10 30];
mAP_hkmaj = [0.168116 0.180195 0.180662];

hold on;
plot(ransacThr_ref, mAP_ref, "Color", "red", "Marker", "o", "LineStyle", "--", "LineWidth", 4);
plot(ransacThr_hkmaj, mAP_hkmaj, "Color", "blue", "Marker", "d", "LineStyle", "--", "LineWidth", 4);
hold off;
grid on;

xlabel("RANSAC re-projection threshold");
ylabel("Mean Average Precision");

legend("Reference", "Best HKMaj", "location", "east");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 10);

print(h, "-dpng", "-color", "map_test_gv_ransacThr.png");

% 4. Combining different detectors and descriptors - Rotation and Scaling

hold on;
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_features_scale", 55, "blue", "-");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_features_none", 55, "green", "--");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_features_rotation", 55, "black", "-.");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_features_rotation_scale", 55, "red", ":");
hold off;
grid on;

legend("Pure scale", "Planar geometry", "Pure rotation", "Rotation + Scale")

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 12);

print(h, "-dpng", "-color", "map_test_features.png");

% 4. Combining different detectors and descriptors - Using more keypoints

h = figure(1);
H = 5;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

hold on;
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_features_none", 55, "c", "-");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_features_rotation_scale_morekpts", 55, "magenta", "-");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_features_none_10pct", 55, "blue", "-");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_features_none_20pct", 55, "green", "--");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_features_none_30pct", 55, "black", ":");
plot_overall_performance("/home/andresf/oxford_buildings_dataset/ranked_files_features_scale_morekpts", 55, "red", "-");
hold off;
grid on;

legend("None, HARRIS, BRIEF", "Rotation + Scale, BRISK (10 pixels threshold), BRISK", "None, AGAST (10% more keypoints), BRIEF", "None, AGAST (20% more keypoints), BRIEF", "None, AGAST (30% more keypoints), BRIEF", "Scale, BRISK (10 pixels threshold), BRIEF");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 10);

print(h, "-dpng", "-color", "map_test_features_morekpts.png");

% 5a.

hold on;
plot_overall_performance("/home/andresf/paris_buildings_dataset/ranked_files_vocab_generic", 55, "red", "-");
plot_overall_performance("/home/andresf/paris_buildings_dataset/ranked_files_vocab_orig", 55, "blue", "--");
hold off;
grid on;

legend("Independent set", "Target set");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 10);

print(h, "-dpng", "-color", "map_test_generalization_paris.png");

% 5b.

hold on;
plot_overall_performance("/home/andresf/rome_landmarks_dataset/ranked_files_vocab_generic", 76, "red", "-");
plot_overall_performance("/home/andresf/rome_landmarks_dataset/ranked_files_vocab_orig", 76, "blue", "--");
hold off;
grid on;

legend("Independent set", "Target set");

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 10);

print(h, "-dpng", "-color", "map_test_generalization_rome.png");

% 6. 

h = figure(1);
H = 3;
W = H*4/3;
set(h, "PaperUnits", "inches")
set(h, "PaperOrientation", "portrait");
set(h, "PaperSize", [H,W]);
set(h, "PaperPosition", [0,0,W,H]);

hold on;
plot_overall_performance("/home/andresf/rome_landmarks_dataset/ranked_files_vocab_16384_deep", 76, "b", "-.");
plot_overall_performance("/home/andresf/rome_landmarks_dataset/ranked_files_vocab_16384_flat", 76, "r", "-");
hold off;

legend("Depth 14, Branching 2", "Depth 7, Branching 4")

fontName = findall(h, "-property", "FontName");
set(fontName, "FontName", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf");
fontSize = findall(h, "-property", "FontSize");
set(fontSize, "FontSize", 12);

print(h, "-dpng", "-color", "map_test_generalization_rome_brisk.png");

