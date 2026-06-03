true_theta = [1.2 1 1 0.1 0.5 0.5];
L = 30;
process_model = process_model2();
param_component = 5; % Should be manually adjusted based on parameter component to be profiled
amount_of_end_points = 4;

switch param_component
    case 1
        param_name = "D";
        param_char = 'D';
        parameter_bounds = [0.2,2.2];
    case 2
        param_name = "mu";
        param_char = '\mu';
        parameter_bounds = [0.2,1.8];
    case 3
        param_name = "K";
        param_char = 'K';
        parameter_bounds = [0.2,1.8];
    case 4
        param_name = "r";
        param_char = 'r';
        parameter_bounds = [0.00005,1];
    case 5
        param_name = "lambda";
        param_char = '\lambda';
        parameter_bounds = [0.00005,1];
    case 6
        param_name = "m0";
        param_char = 'm_0';
        parameter_bounds = [0.2,0.8];
end

alpha = 0.05;
lower_bound_ci = exp(-chi2inv(1 - alpha,1)/2);
for K_i = [21,51]
    K_i
    for j = 1:2
        T = j*100 % Can be adjusted based on desired temporal end point (j*...)
        observation_domain = [0,L;0,T];
        for evaluation_amount_x = [25,50] % Can be adjusted based on desired spatial resolution of the data
            evaluation_amount_x 
            for base_resolution = [4,8] % Can be adjusted based on desired temporal resolution of the data
                evaluation_amount_t_per_interval = j*base_resolution
                seed = rng(100);
                full_data = observed_data(observation_function(process_model,observation_domain,true_theta, amount_of_end_points,evaluation_amount_x,evaluation_amount_t_per_interval),true_theta);
                fig = figure('WindowState','maximized');
                tiledlayout(2,2);
                
                for i = 1:4
                    i
                    amount_of_time_points = (length(full_data(1,:)) - 1)*i/amount_of_end_points + 1;
                    data = full_data(:,1:amount_of_time_points);
                
                    [npl, ll, conf_int, optimal_parameters, runtime, full_domain] = profile_likelihood(param_component, observation_domain, parameter_bounds, data, i, ...
                        K_i,evaluation_amount_x,evaluation_amount_t_per_interval);
                
                    nexttile
                    plot(full_domain, npl,'-b','DisplayName','Computed NPL','LineWidth',2);
                    xlabel(param_char);
                    ylabel('NPL');
                    ylim([0,1.05])
                    title(['End Point ', num2str(i*T/4), ', K_i = ', num2str(K_i) ', Runtime:', num2str(runtime),' s']);
                    xline(true_theta(param_component),'--','DisplayName','True Value')
                    part_of_ci = find(npl >= lower_bound_ci);
                    not_part_of_ci = find(npl < lower_bound_ci);
                    index = 1;
                    no_ci_label = true;
                    while ~isempty(part_of_ci)
                        current_min = part_of_ci(1);
                        not_part_of_ci = not_part_of_ci(not_part_of_ci > current_min);
                
                        if isempty(not_part_of_ci)
                            current_max = length(full_domain);
                        else
                            current_max = not_part_of_ci(1);
                        end
                        part_of_ci = part_of_ci(part_of_ci > current_max);
                        index = index + 1;
                
                        if no_ci_label
                            xregion([full_domain(current_min); full_domain(current_max)],'DisplayName','CI (95%)')
                            no_ci_label = false;
                        else
                            xregion([full_domain(current_min); full_domain(current_max)],'HandleVisibility','off')
                        end
                    end
                    legend
                end
                
                print(fig,strcat("E:\Apps\MATLAB\R2024b\Projects\Bachelors Thesis\All Code and Logs - Final\Crossley's Full Model\Plots\","Crossley (",param_name, num2str(K_i), ", N_x ", num2str(evaluation_amount_x), ", N_t ", num2str(evaluation_amount_t_per_interval), ", T ", num2str(T), ", REWORKED MODEL).png"),"-dpng") % Should be manually adjusted based on the desired save directory for the plots (up to ...\Plots); modified model can be adjusted in script "f.m"
                close(fig);
            end
        end
    end
end
clear