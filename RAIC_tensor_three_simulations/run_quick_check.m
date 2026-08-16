%RUN_QUICK_CHECK Run a small smoke test before the paper-scale experiment.
clear;
clc;
close all;

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
run_three_simulations('quick');
