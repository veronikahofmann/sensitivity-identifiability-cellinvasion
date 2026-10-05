function data = observed_data(observed_function, true_theta)

arguments (Input)
    observed_function
    true_theta
end

arguments (Output)
    data
end

%------------------------------------------------------------------------
dimensions = size(observed_function);
true_variance = 0.05;
errors = zeros(dimensions);
K = true_theta(2);
pd = makedist('Normal','mu',0,'sigma',true_variance);

for j = 1:dimensions(2)
    for i = 1:dimensions(1)
        t = truncate(pd,-observed_function(i,j), K - observed_function(i,j));
        errors(i,j) = random(t);
    end
end

%errors = sqrt(true_variance)*randn(dimensions);
data = observed_function + errors;
end