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
alpha = 0.5;

grid.L = domain(1,2);
grid.dx = 0.5;
x_star = (0:grid.dx:grid.L)';
grid.Nx = length(x_star);

K = theta(3);
m0_val = theta(6);

u0 = K*double(x_star < alpha);
m0 = m0_val * double(x_star >= alpha);
y0 = [u0; m0];

t_span = [0 domain(2,2)];
[t_star, y] = ode15s(@(t,y) f(t, y, theta, grid), t_span, y0);

u = y(:,1:grid.Nx);          
m = y(:,grid.Nx+1:end);

u_at_t = interp1(t_star, u, t, 'pchip', 'extrap');  
m_at_t = interp1(t_star, m, t, 'pchip', 'extrap');

u_eval = interp1(x_star, u_at_t', x, 'pchip','extrap');
m_eval = interp1(x_star, m_at_t', x, 'pchip','extrap');

pde_eval = [u_eval; m_eval]; %NOTE: [u(x_1, t_1), u(x_1, t_2), ...; u(x_2, t_1), u(x_2, t_2), ...; ...; m(x_1, t_1), m(x_1, t_2), ...; ...;]
end