function [npl, ll, conf_int, optimal_parameters, runtime, full_domain] = profile_likelihood(param_index, domain, bounds, data, end_time, K_i,evaluation_amount_x,evaluation_amount_t_per_interval)
%pl Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    param_index
    domain
    bounds
    data
    end_time
    K_i
    evaluation_amount_x
    evaluation_amount_t_per_interval
end

arguments (Output)
    npl
    ll
    conf_int
    optimal_parameters
    runtime
    full_domain
end

%------------------------------------------------------------------------
tic;
alpha = 0.05;

assumed_variance = 0.05;
process_model = process_model2();
number_parameters = 6;
res = @(theta) data - observation_function(process_model,domain,theta, end_time,evaluation_amount_x,evaluation_amount_t_per_interval);
likelihood = @(theta) exp(-1/(2*assumed_variance)*norm(res(theta),'fro')^2);

profiled_values = linspace(bounds(1),bounds(2),K_i);
optimal_parameters = zeros(K_i,6);
pl = zeros(1,K_i);
ll = zeros(1,K_i);

p0 = [1 1 1 0.5 0.5 0.3]; 
options = optimoptions('lsqnonlin', 'Display', 'off');

for i = 1:K_i
    fixed_val = profiled_values(i)

    if param_index == 2
        p0 = [1 fixed_val fixed_val*2 0.5 0.5 0.3*fixed_val]; 
    elseif param_index == 3
        p0 = [1 fixed_val fixed_val 0.5 0.5 0.3*fixed_val]; 
    end

    lb = zeros(1,number_parameters);
    ub = Inf(1,number_parameters);
    A = [];
    b = [];
    Aeq = [zeros(1,param_index-1) 1 zeros(1,6-param_index)];
    beq = fixed_val;
    nonlcon = @movement_condition;

    optimal_parameter = lsqnonlin(res, p0, lb, ub, A, b, Aeq, beq, nonlcon, options);
    pl(i) = likelihood(optimal_parameter);
    ll(i) = norm(res(optimal_parameter))^2;
    optimal_parameters(i, :) = optimal_parameter;
end

if all(pl <= 0)
    npl = ones(1,K_i);
else
    npl = pl / max(pl);
end

zoom_factor = 50;
full_domain = linspace(bounds(1),bounds(2),(K_i-1)*zoom_factor+1);
npl = interp1(profiled_values,npl,full_domain,'pchip');
ll = interp1(profiled_values,ll,full_domain,'pchip');
optimal_parameters = interp1(profiled_values,optimal_parameters,full_domain,'pchip');

lower_bound_ci = exp(-chi2inv(1 - alpha,1)/2);
conf_int = [full_domain(find(npl >= lower_bound_ci, 1, 'first')),full_domain(find(npl >= lower_bound_ci, 1, 'last'))];
runtime = toc;
end

function [c,ceq] = movement_condition(p)
precision = 0.2;
c(1) = p(2)*p(6) - (1-precision)*p(3);
c(2) = precision*p(3) - p(2)*p(6);
ceq = [];
end