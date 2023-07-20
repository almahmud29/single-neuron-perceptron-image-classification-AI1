clear
close all

load data.mat;


% define useful parameter

num_tr = 500; % sample of training data
num_test = 2000; % sample of testting data
weight_int = rand(1, 3)';
iteration = 50;
eta = 1;

weight = weight_int;

% For training purpose
for ii = 1: iteration
    shuffle_seq = randperm(2500);
    data_shuffled_tr = data(:, shuffle_seq);

    for jj = 1:num_tr
        x = [1; data_shuffled_tr(1:2, jj)];
        d = [data_shuffled_tr(3, jj)];
        v = weight'*x;

        y = my_activation(v);

        e(jj) = d - y;

        % steepest descent method
        weight = weight + eta*(d-y)*x;
    end

    MSE(ii) = mean(e.^2);
end

%  this part is for testing purpose

%shuffle_seq = randperm(2500);
data_shuffled_test = data(:, shuffle_seq);

for ii = 501:2500
    input(:,ii) = [1; data_shuffled_test(1:2, ii)];
    v_initial = weight_int'*input(:,ii);
    v_final = weight'*input(:,ii);

    output_int(ii) = my_activation(v_initial);
    output_opt(ii) = my_activation(v_final);

end

% display the output

for ii = 1:2500
    x = input(2:3, ii);
    y = output_int(ii);

    if y==1
        plot(x(1), x(2), 'r*', 'Linewidth', 3, 'MarkerSize', 9);
    else
        plot(x(1), x(2), 'b*', 'Linewidth', 3, 'MarkerSize', 9);
    end
    hold on;
end

figure
for ii = 1:2500
    x = input(2:3, ii);
    y = output_opt(ii);

    if y==1
        plot(x(1), x(2), 'r*', 'Linewidth', 3, 'MarkerSize', 9);
    else
        plot(x(1), x(2), 'b*', 'Linewidth', 3, 'MarkerSize', 9);
    end
    hold on;
end