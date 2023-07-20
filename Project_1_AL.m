clc
clear all
%% Generate double half moon pattern

rad = 10; % radius of half moon
width = 6; % width of half moon
dist = 0.5;

num_samp = 2500; % number of samples

aa = rand(2, num_samp/2);

radius = (rad-width/2) + width*aa(1, :);

% class 1

theta1 = pi*aa(2,:) + 20*pi/180;
x1 = radius.*cos(theta1);
y1 = radius.*sin(theta1);
label1 = 1*ones(1, length(x1)); % label for class 1

% class 2

theta2 = pi*aa(2,:) - 20*pi/180;
x2 = radius.*cos(-theta2) + rad;
y2 = radius.*sin(-theta2) - dist;
label2 = -1*ones(1, length(x2)); % label for class 2

% adding data together

data = [x1, x2;
    y1, y2;
    label1, label2];

% display the data
for ii = 1:2500
    x = [1; data(1:2, ii)];
    y = data(3, ii);

    if y==1
        plot(x(2), x(3), 'r*', 'LineWidth', 2, 'MarkerSize', 9);
    elseif y==-1
        plot(x(2), x(3), 'b*', 'LineWidth', 2, 'MarkerSize', 9);
    end
    hold on;
end

hh = xlabel('x_1');
set(hh, 'FontSize', 26, 'FontWeight', 'bold');

hold on;
hh = ylabel('x_2');
set(hh, 'FontSize', 26, 'FontWeight', 'bold');

set(gca, 'FontSize', 26, 'FontWeight', 'bold');

grid;

save data data


%% define useful parameter

num_tr = 500; % sample of training data
num_test = 2000; % sample of testting data

%% Initialize weight vector

weight_int = rand(1, 3)';
iteration = 50;
weight = weight_int;


%% 

shuffle_seq = randperm(2500);
data_shuffled_tr = data(:, shuffle_seq);
% For training purpose
for ii = 1: iteration
%     shuffle_seq = randperm(2500);
%     data_shuffled_tr = data(:, shuffle_seq);
    
    %  Create Jacobian Matrix
    for j = 1:num_tr
        x = [1; data_shuffled_tr(1:2, j)];
        for i = 1:length(x)
            J(j,i) = [-x(i)];
        end
    end

    % Construct Neural Network
    for jj = 1:num_tr
        x = [1; data_shuffled_tr(1:2, jj)];
        d = [data_shuffled_tr(3, jj)];

        v = weight'*x;

        y = my_activation(v);
        
        e(jj) = d - 1.*weight(1) - weight(2).*x(2) - weight(3).*x(3);
    end


    % Gauss Newton method

    JJ = J'*e';
    JJJ = inv(J'*J+eye(3));
    JJJJ = JJJ*JJ;
    weight = weight - JJJJ; % update weight

    MSE(ii) = mean(e.^2);
end

%  this part is for testing purpose

data_shuffled_test = data(:, shuffle_seq);

%% Plot train data 

for ii = 1:500
    input(:,ii) = [1; data_shuffled_test(1:2, ii)];
    v_initial = weight_int'*input(:,ii);
    v_final = weight'*input(:,ii);

    output_int(ii) = my_activation(v_initial);
    output_opt(ii) = my_activation(v_final);

end

% display the output
figure
for ii = 1:500
    x = input(2:3, ii);
    y = output_int(ii);

    if y==1
        plot(x(1), x(2), 'r*', 'Linewidth', 3, 'MarkerSize', 2);
    else
        plot(x(1), x(2), 'b*', 'Linewidth', 3, 'MarkerSize', 2);
    end
    hold on;
end

hh = xlabel('x_1');
set(hh, 'FontSize', 12, 'FontWeight', 'bold');

hold on;
hh = ylabel('x_2');
set(hh, 'FontSize', 12, 'FontWeight', 'bold');

set(gca, 'FontSize', 12, 'FontWeight', 'bold');

grid;

%%

% display the output
figure
for ii = 1:500
    x = input(2:3, ii);
    y = output_opt(ii);

    if y==1
        plot(x(1), x(2), 'r*', 'Linewidth', 3, 'MarkerSize', 2);
    else
        plot(x(1), x(2), 'b*', 'Linewidth', 3, 'MarkerSize', 2);
    end
    hold on;
end


hh = xlabel('x_1');
set(hh, 'FontSize', 12, 'FontWeight', 'bold');

hold on;
hh = ylabel('x_2');
set(hh, 'FontSize', 12, 'FontWeight', 'bold');

set(gca, 'FontSize', 12, 'FontWeight', 'bold');

grid;
%%

for ii = 501:2500
    input(:,ii) = [1; data_shuffled_test(1:2, ii)];
    v_initial = weight_int'*input(:,ii);
    v_final = weight'*input(:,ii);

    output_int(ii) = my_activation(v_initial);
    output_opt(ii) = my_activation(v_final);

end
%%
% display the output
figure
for ii = 501:2500
    x = input(2:3, ii);
    y = output_int(ii);

    if y==1
        plot(x(1), x(2), 'r*', 'Linewidth', 3, 'MarkerSize', 2);
    else
        plot(x(1), x(2), 'b*', 'Linewidth', 3, 'MarkerSize', 2);
    end
    hold on;
end


hh = xlabel('x_1');
set(hh, 'FontSize', 12, 'FontWeight', 'bold');

hold on;
hh = ylabel('x_2');
set(hh, 'FontSize', 12, 'FontWeight', 'bold');

set(gca, 'FontSize', 12, 'FontWeight', 'bold');

grid;
%%
figure
for ii = 501:2500
    x = input(2:3, ii);
    y = output_opt(ii);

    if y==1
        plot(x(1), x(2), 'r*', 'Linewidth', 3, 'MarkerSize', 2);
    else
        plot(x(1), x(2), 'b*', 'Linewidth', 3, 'MarkerSize', 2);
    end
    hold on;
end


hh = xlabel('x_1');
set(hh, 'FontSize', 12, 'FontWeight', 'bold');

hold on;
hh = ylabel('x_2');
set(hh, 'FontSize', 12, 'FontWeight', 'bold');

set(gca, 'FontSize', 12, 'FontWeight', 'bold');

grid;

%% MSE vs Iteration

figure
for ii = 1:iteration
    x = ii;
    y = MSE(ii);
        plot(x, y, 'k*', 'Linewidth', 3, 'MarkerSize', 2);
    hold on;
end


hh = xlabel('iteration');
set(hh, 'FontSize', 12, 'FontWeight', 'bold');

hold on;
hh = ylabel('MSE');
set(hh, 'FontSize', 12, 'FontWeight', 'bold');

set(gca, 'FontSize', 12, 'FontWeight', 'bold');

grid;