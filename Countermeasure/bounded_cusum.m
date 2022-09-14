function [alarms_upper, alarms_lower, uppersum, lowersum] = bounded_cusum(residuals, control_limit,minimum_mean_shift, residuals_mean, residuals_std)
    %this function is based on the equations available at https://en.mathworks.com/help/signal/ref/cusum.html
    %We modify the cusum equations to put an upper bound and a lower 
    %bound on the upper and lower cumulative sums to
    %decrease the false alarms
    threshold = control_limit*residuals_std;
    cusum_bound = (control_limit+2)*residuals_std;
    uppersum = zeros(size(residuals));
    lowersum = zeros(size(residuals));
    for i=1:length(residuals)
      if i == 1
         %i = 1 case
        uppersum(i) = 0;
        lowersum(i) = 0;
      else
        %i > 1 case
        uppersum(i) = min(cusum_bound, u_sum(residuals, i, uppersum, residuals_mean, minimum_mean_shift, residuals_std));
        lowersum(i) = max(-cusum_bound , l_sum(residuals, i, lowersum, residuals_mean, minimum_mean_shift, residuals_std));
      end
    end
    alarms_upper = find(uppersum > threshold);
    alarms_lower = find(lowersum < -threshold);
    
end

function sum = u_sum(residuals, i, upper_sum, residual_mean, minimum_mean_shift, residual_standard_deviation)
    sum =  max(0,upper_sum(i-1) + residuals(i) - residual_mean - 0.5*minimum_mean_shift*residual_standard_deviation);
end

function sum = l_sum(residuals, i, lower_sum, residual_mean, minimum_mean_shift, residual_standard_deviation)
    sum =  min(0,lower_sum(i-1) + residuals(i) - residual_mean + 0.5*minimum_mean_shift*residual_standard_deviation);
end