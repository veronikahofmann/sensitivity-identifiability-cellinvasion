function dydt = f(~,y,theta,grid)
%pl Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    ~
    y
    theta
    grid
end

arguments (Output)
    dydt
end

%------------------------------------------------------------------------
    D = theta(1);
    mu = theta(2);
    K = theta(3);
    r = theta(4);
    lambda = theta(5);
    
    u = y(1:grid.Nx);          
    m = y(grid.Nx+1:end);      

    u_ext = [u(1); u; u(end)];   
    m_ext = [m(1); m; m(end)];

    capac = (u_ext + mu * m_ext) / K;   
    log_term = 1 - capac;

    i = 2 : grid.Nx + 1; 

    L_left = log_term(i-1) + log_term(i);
    L_mid = log_term(i-1) + 2*log_term(i) + log_term(i+1);
    L_right = log_term(i) + log_term(i+1);

    u_left = u_ext(i-1);
    u_mid = u_ext(i);
    u_right = u_ext(i+1);

    cap_left = capac(i-1);
    cap_mid = capac(i);
    cap_right = capac(i+1);

    du_dt = D/(2*grid.dx^2) * ( ...
          L_left.*u_left ...
        - L_mid.*u_mid  ...
        + L_right.*u_right ...
        + (u_left + u_mid).*cap_left ...
        - (u_left + 2*u_mid + u_right).*cap_mid ...
        + (u_mid + u_right).*cap_right ) ...
        + r*u_mid.*log_term(i);

    dm_dt = -lambda*u.*m; %Current Status: Crossley's Original Model (Modified Model: -lambda/K*u.*m)

    dydt = [du_dt; dm_dt];
end
