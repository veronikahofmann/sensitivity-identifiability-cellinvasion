full_plot_directory = "E:\Apps\MATLAB\R2024b\Projects\Bachelors Thesis\All Code and Logs - Final\Crossley's Full Model\New Plots (Crossley, More Thetas)\Data Showcase";
if isfolder(full_plot_directory) ~= true
    mkdir(full_plot_directory)
end

width_px = 3840;
height_px = 2160;
dpi = 200; 
width_in = width_px / dpi;
height_in = height_px / dpi;

process_model = process_model2();
observation_domain = [0,30;0,100];
amount_of_end_points = 4;
evaluation_amount_x = 20;
evaluation_amount_t_per_interval = 6;

true_thetas = [
    1   1   1   0.1 0.5 0.5; %Original Theta
    0.2 1   1   0.1 0.5 0.5; %Theta 2: Reduced D
    1   0.4 0.4 0.1 0.5 0.5; %Theta 3: Reduced mu and K but equal
    0.5 1   1   0.1 1   0.5; %Theta 4: Reduced D and increased lambda
    1   1   1   0.5 0.1 0.5; %Theta 5: Reduced lambda and increased r
    ];

for theta_index = 1:size(true_thetas,1)
    true_theta = true_thetas(theta_index,:);
    full_data = observation_function(process_model,observation_domain,true_theta, amount_of_end_points,evaluation_amount_x,evaluation_amount_t_per_interval);
    
    %fig = figure('WindowState','maximized');
    fig = figure('Visible','off');
    %tiledlayout(2,1);
    
    x = linspace(0,30,evaluation_amount_x);
    t = linspace(0,200,evaluation_amount_t_per_interval*4 + 1);
    
    %[T,X] = meshgrid(x,t);
    u_data = full_data(1:(end/2),:);
    m_data = full_data((end/2 + 1):end,:);
    
    surf(t,x,u_data);
    xlabel('t');
    ylabel('x')
    zlabel('density');
    zlim([0,1.05])
    title('u(x,t)');
    n = 100; 
    a = linspace(0,1,n)';
    w = [204,204,255]/255;
    p = [75 0 130]/255;
    CM1 = (1-a)*w + a*p;
    colormap(CM1)
    set(fig, 'PaperPositionMode', 'manual');
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperPosition', [0, 0, width_in, height_in]);
    fontsize(fig,32,"pixels")
    print(fig,strcat(full_plot_directory,"\Crossley (u_full, Theta ", num2str(theta_index),").png"),"-dpng", ['-r' num2str(dpi)])
    
    %fig = figure('WindowState','maximized');
    fig = figure('Visible','off');
    
    surf(t,x,m_data)
    xlabel('t');
    ylabel('x')
    zlabel('density');
    zlim([0,1.05])
    title('m(x,t)');
    n = 100; 
    a = linspace(0,1,n)';
    w = [255,238,140]/255;
    p = [186 142 35]/255;
    CM2 = (1-a)*w + a*p;
    colormap(CM2)
    set(fig, 'PaperPositionMode', 'manual');
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperPosition', [0, 0, width_in, height_in]);
    fontsize(fig,32,"pixels")
    print(fig,strcat(full_plot_directory,"\Crossley (m_full, Theta ", num2str(theta_index),").png"),"-dpng", ['-r' num2str(dpi)])
end
clear

