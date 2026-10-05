function pde_eval = pde_evaluation(x,t,theta, domain)
%PDE_EVALUATION Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    x
    t
    theta
    domain
end

arguments (Output)
    pde_eval
end

%------------------------------------------------------------------------
sigma = 2;

grid.L = domain(1,2);
grid.dx = 0.5;
x_star = (0:grid.dx:grid.L)';
grid.Nx = length(x_star);

K = theta(3);
m0_val = theta(6);
omega = theta(7);

n0 = K*double(x_star <= sigma - omega) + K*exp(1 - 1./(1- ((x_star - sigma + omega)./omega).^2)).*double(sigma - omega < x_star & x_star < sigma);
m0 = m0_val * double(x_star >= sigma) + m0_val*(1 - n0/K).*double(sigma - omega <= x_star & x_star < sigma);
y0 = [n0; m0];

final_t = max(t(:));
t_span = [0 min(domain(2,2),final_t)];

N = 2*grid.Nx;
JPattern = spalloc(N,N,10*grid.Nx);

for i = 1:grid.Nx

    % n_i equation
    JPattern(i,i) = 1;

    if i > 1
        JPattern(i,i-1) = 1;
    end

    if i < grid.Nx
        JPattern(i,i+1) = 1;
    end

    JPattern(i,grid.Nx+i) = 1;

    if i > 1
        JPattern(i,grid.Nx+i-1) = 1;
    end

    if i < grid.Nx
        JPattern(i,grid.Nx+i+1) = 1;
    end

    % m_i equation
    JPattern(grid.Nx+i,i) = 1;
    JPattern(grid.Nx+i,grid.Nx+i) = 1;
end


opts = odeset('JPattern',JPattern);

%tic
[t_star, y] = ode15s(@(t,y) f(t, y, theta, grid), t_span, y0, opts);
%ode_time = toc;

%fprintf('\nODE time: %.3f s\n',ode_time);
%fprintf('ODE points: %d\n',length(t_star));
%fprintf('Average dt: %.6g\n',mean(diff(t_star)));
%fprintf('Minimum dt: %.6g\n',min(diff(t_star)));
%fprintf('Maximum dt: %.6g\n',max(diff(t_star)));

n = y(:,1:grid.Nx);          
m = y(:,grid.Nx+1:end);

n_at_t = interp1(t_star, n, t, 'pchip', 'extrap');  
m_at_t = interp1(t_star, m, t, 'pchip', 'extrap');

n_eval = interp1(x_star, n_at_t', x, 'pchip','extrap');
m_eval = interp1(x_star, m_at_t', x, 'pchip','extrap');

pde_eval = [n_eval; m_eval]; %NOTE: [n(x_1, t_1), n(x_1, t_2), ...; n(x_2, t_1), n(x_2, t_2), ...; ...; m(x_1, t_1), m(x_1, t_2), ...; ...;]
end