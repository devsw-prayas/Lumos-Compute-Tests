clear; clc; close all;

lambda = linspace(-6, 6, 4000);

% Parameters
n     = 2;
mu    = 0;
sigma = 1;
omega = 3;

x = (lambda - mu) / sigma;

Phi = (1 / sqrt(2^n * factorial(n) * sqrt(pi) * sigma)) .* ...
      hermiteH(n, x) .* ...
      exp(-x.^2 / 2) .* ...
      exp(1i * omega * (lambda - mu));

figure;
plot3(lambda, real(Phi), imag(Phi), 'LineWidth', 1.3);
grid on;
xlabel('\lambda');
ylabel('Re[\Phi(\lambda)]');
zlabel('Im[\Phi(\lambda)]');
title('GHGSF in 3D: (\lambda, Re, Im)');
view(45, 25);

theta = unwrap(angle(Phi));

figure;
plot3(lambda, abs(Phi).*cos(theta), abs(Phi).*sin(theta), 'LineWidth', 1.3);
grid on;
xlabel('\lambda');
ylabel('Magnitude · cos(phase)');
zlabel('Magnitude · sin(phase)');
title('Explicit polar reconstruction in 3D');
view(45, 30);
