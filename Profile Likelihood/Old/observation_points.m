function [evaluation_points_x,evaluation_points_t] = observation_points(domain,end_time,evaluation_amount_x,evaluation_amount_t_per_interval)
%OBSERVATION_POINTS Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    domain
    end_time
    evaluation_amount_x
    evaluation_amount_t_per_interval
end

arguments (Output)
    evaluation_points_x
    evaluation_points_t 
end

final_times = 4;

domain_x = domain(1,:);
domain_t = domain(2,:);

evaluation_points_x = linspace(domain_x(1), domain_x(2), evaluation_amount_x);
evaluation_points_t = linspace(domain_t(1), end_time*domain_t(2)/final_times, evaluation_amount_t_per_interval*end_time + 1);
end