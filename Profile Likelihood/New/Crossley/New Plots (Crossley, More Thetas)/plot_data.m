true_thetas = [
    1   1   1   0.1 0.5 0.5;
    0.2 1   1   0.1 0.5 0.5;
    1   0.4 0.4 0.1 0.5 0.5;
    0.5 1   1   0.1 1   0.5;
    1   1   1   0.5 0.1 0.5;
];


% Directories
results_directory = fullfile(pwd, "Results");
base_plot_directory = fullfile(pwd, "Plots");
if ~isfolder(base_plot_directory)
    mkdir(base_plot_directory);
end


% Figure Settings
width_px = 3840;
height_px = 2160;
dpi = 200;

width_in = width_px / dpi;
height_in = height_px / dpi;

% Confidence Interval
alpha = 0.05;
lower_bound_ci = exp(-chi2inv(1-alpha,1)/2);


% Parameter Metadata
param_names = ["D","mu","K","r","lambda","m0"];
param_chars = {
    'D'
    '\mu'
    'K'
    'r'
    '\lambda'
    'm_0'
};

% Find existing results files
files = dir(fullfile( ...
    results_directory, ...
    'Theta*_EP*.mat'));

disp("RESULT FILE DISCOVERY");

disp("Results directory:");
disp(results_directory);

fprintf("Found %d result files.\n", length(files));

if isempty(files)
    error("No result files found in: %s", results_directory);
end


% Extract combinations
config_names = strings(length(files),1);
for f = 1:length(files)
    filename = files(f).name;
    config_names(f) = regexprep(filename, '_EP\d+\.mat$','');
end

% Remove duplicates
unique_configs = unique(config_names);

fprintf("Found %d unique configurations.\n", length(unique_configs));

disp("STARTING PLOT GENERATION");

% Process each existing combination
for c = 1:length(unique_configs)
    config_name = unique_configs(c);
    disp(" ");
    disp("------------------------------------------------");
    disp("Processing: " + config_name);
    disp("------------------------------------------------");
    tokens = regexp(config_name, ...
        '^Theta(\d+)_(.+)_Ki(\d+)_T(\d+)_Nx(\d+)_Nt(\d+)$', ...
        'tokens', ...
        'once');

    if isempty(tokens)
        warning("Could not understand configuration name: %s", config_name);
        continue;
    end

    theta_index = str2double(tokens{1});
    param_name = tokens{2};
    K_i = str2double(tokens{3});
    T = str2double(tokens{4});
    evaluation_amount_x = str2double(tokens{5});
    evaluation_amount_t_per_interval = str2double(tokens{6});


    if theta_index < 1 || theta_index > size(true_thetas,1)
        warning("Invalid theta index %d in %s", theta_index, config_name);
        continue;
    end

    true_theta = true_thetas(theta_index,:);

    param_component = find(param_names == param_name, 1);

    if isempty(param_component)
        warning("Unknown parameter '%s' in %s", param_name, config_name);
        continue;
    end

    param_char = param_chars{param_component};

    %Create output directories
    theta_plot_directory = fullfile(base_plot_directory, sprintf("Theta %d",theta_index));
    if ~isfolder(theta_plot_directory)
        mkdir(theta_plot_directory);
    end

    param_plot_directory = fullfile(theta_plot_directory, param_name);
    if ~isfolder(param_plot_directory)
        mkdir(param_plot_directory);
    end

    %Create figure
    fig = figure('Visible','off', 'Color','white');
    tiledlayout(2,2);

    %Configure plot
    for i = 1:4
        endpoint_filename = sprintf(...
            'Theta%d_%s_Ki%d_T%d_Nx%d_Nt%d_EP%d.mat', ...
            theta_index, ...
            param_name, ...
            K_i, ...
            T, ...
            evaluation_amount_x, ...
            evaluation_amount_t_per_interval, ...
            i);

        endpoint_filepath = fullfile(results_directory, endpoint_filename);

        % Check whether endpoint actually exists
        if ~isfile(endpoint_filepath)
            disp("  Missing endpoint " + i + ": " + endpoint_filename);
            continue;
        end

        disp("  Loading endpoint " + i + ": " + endpoint_filename);

        result = load(endpoint_filepath);

        required_fields = {
            'full_domain'
            'npl'
            'runtime'
        };

        missing_fields = required_fields(~isfield(result,required_fields));

        if ~isempty(missing_fields)
            warning("Skipping %s because required fields are missing.", endpoint_filename);
            continue;
        end

        nexttile;

        plot(result.full_domain, result.npl, ...
            '-b', 'DisplayName','Computed NPL', 'LineWidth',2);

        hold on;

        % Axis labels
        xlabel(param_char);
        ylabel('NPL');
        ylim([0,1.05]);

        % Endpoint title
        title(sprintf( ...
            'End Point %d, Runtime: %.2f s', ...
            i*T/4, ...
            result.runtime));

        % True parameter value
        xline(true_theta(param_component), '--', 'DisplayName','True Value');

        %Confidence Interval Calculations
        part_of_ci = find(result.npl >= lower_bound_ci);
        not_part_of_ci = find(result.npl < lower_bound_ci);
        no_ci_label = true;

        while ~isempty(part_of_ci)
            current_min = part_of_ci(1);
            not_part_of_ci = not_part_of_ci(not_part_of_ci > current_min);

            if isempty(not_part_of_ci)
                current_max = length(result.full_domain);
            else
                current_max = not_part_of_ci(1);
            end

            part_of_ci = part_of_ci(part_of_ci > current_max);

            if no_ci_label
                xregion(...
                    [result.full_domain(current_min); ...
                    result.full_domain(current_max)], ...
                    'DisplayName','CI (95%)');
                no_ci_label = false;
            else
                xregion( ...
                    [result.full_domain(current_min); ...
                    result.full_domain(current_max)], ...
                    'HandleVisibility','off');
            end
        end

        legend;
        hold off;
    end

    %Formatting
    set(fig, 'Units','pixels', 'Position',[100 100 width_px height_px]);
    set(fig, 'PaperUnits','inches', 'PaperPosition',[0 0 width_in height_in]);
    fontsize(fig, 32, "pixels");

    %Saving Procedure
    output_filename = sprintf( ...
        'Crossley (%s%d, N_x %d, N_t %d, T %d).png', ...
        param_name, ...
        K_i, ...
        evaluation_amount_x, ...
        evaluation_amount_t_per_interval, ...
        T);

    output_file = fullfile(param_plot_directory, output_filename);
    disp("  Saving: " + output_filename);
    print(fig, output_file, '-dpng', printf('-r%d',dpi));

    close(fig);
end

disp(" ");
disp("==============================================");
disp("ALL EXISTING RESULT CONFIGURATIONS HAVE BEEN PLOTTED.");
disp("==============================================");

clear