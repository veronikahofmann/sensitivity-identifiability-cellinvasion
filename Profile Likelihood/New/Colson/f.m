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
    m_max = theta(2);
    K = theta(3);
    r = theta(4);
    lambda = theta(5);
    
    n = y(1:grid.Nx);          
    m = y(grid.Nx+1:end);   

    n_ext = [n(1); n; n(end)];   
    m_ext = [m(1); m; m(end)];

    capac = m_ext / m_max;   
    log_term = 1 - capac;

    i = 2 : grid.Nx + 1; 

    L_left = log_term(i-1) + log_term(i);
    L_mid = log_term(i-1) + 2*log_term(i) + log_term(i+1);
    L_right = log_term(i) + log_term(i+1);

    n_left = n_ext(i-1);
    n_mid = n_ext(i);
    n_right = n_ext(i+1);

    dn_dt = D/(2*grid.dx^2) * ( ...
          L_left.*n_left ...
        - L_mid.*n_mid  ...
        + L_right.*n_right) ...
        + r*n_mid.*(1-n_mid/K);

    dm_dt = -lambda*n.*m;

    dydt = [dn_dt; dm_dt];
end
