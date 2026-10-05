function observed_function = observation_function(process_model, domain, theta, end_time,evaluation_amount_x,evaluation_amount_t_per_interval)

arguments (Input)
    process_model
    domain
    theta
    end_time
    evaluation_amount_x
    evaluation_amount_t_per_interval
end

arguments (Output)
    observed_function
end

%------------------------------------------------------------------------

[evaluation_points_x,evaluation_points_t] = observation_points(domain,end_time,evaluation_amount_x,evaluation_amount_t_per_interval);
observed_function = process_model(evaluation_points_x, evaluation_points_t, theta,domain);
end