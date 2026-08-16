%RUN_PAPER_EXPERIMENTS Run the intended paper-scale configuration.
clear;
clc;
close all;

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
run_three_simulations('paper');
