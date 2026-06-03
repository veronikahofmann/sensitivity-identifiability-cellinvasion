function process_model = process_model2()
%PROCESS_MODEL Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)

end

arguments (Output)
    process_model
end

%------------------------------------------------------------------------
process_model = @(x,t,theta,domain) pde_evaluation(x,t,theta,domain);
end