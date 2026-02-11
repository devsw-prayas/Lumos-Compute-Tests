lambda = linspace(-5, 5, 2000);
g = exp(-lambda.^2 / 2);

figure;
plot(lambda, g, 'LineWidth', 1.5);
grid on;
xlabel('\lambda');
ylabel('g(\lambda)');
title('Base Gaussian');

mu = 0;        % center
sigma = 1;     % bandwidth

x = (lambda - mu) / sigma;

figure;
plot(lambda, exp(-x.^2 / 2), 'LineWidth', 1.5);
grid on;
title('Gaussian with normalized wavelength');

figure; hold on;
for n = 0:4
    psi = hermiteH(n, x) .* exp(-x.^2 / 2);
    plot(lambda, psi, 'DisplayName', ['n = ', num2str(n)]);
end
legend; grid on;
title('Hermite–Gaussian Functions (unnormalized)');
xlabel('\lambda');
ylabel('\psi_n(\lambda)');

figure; hold on;
for n = 0:4
    normFactor = 1 / sqrt(2^n * factorial(n) * sqrt(pi) * sigma);
    psi = normFactor * hermiteH(n, x) .* exp(-x.^2 / 2);
    plot(lambda, psi, 'DisplayName', ['n = ', num2str(n)]);
end
legend; grid on;
title('Normalized Hermite–Gaussian Functions');

omega = 3;   % modulation frequency
n = 2;

Phi = (1 / sqrt(2^n * factorial(n) * sqrt(pi) * sigma)) * ...
      hermiteH(n, x) .* ...
      exp(-x.^2 / 2) .* ...
      exp(1i * omega * (lambda - mu));

figure;
subplot(2,1,1);
plot(lambda, real(Phi), 'LineWidth', 1.2);
grid on;
title('Real part');

subplot(2,1,2);
plot(lambda, imag(Phi), 'LineWidth', 1.2);
grid on;
title('Imaginary part');
xlabel('\lambda');

figure;
subplot(2,1,1);
plot(lambda, abs(Phi), 'LineWidth', 1.5);
grid on;
title('Magnitude |Φ(\lambda)|');

subplot(2,1,2);
plot(lambda, unwrap(angle(Phi)), 'LineWidth', 1.2);
grid on;
title('Phase θ(\lambda)');
xlabel('\lambda');

figure;
plot(real(Phi), imag(Phi), 'LineWidth', 1.2);
axis equal;
grid on;
xlabel('Re[\Phi]');
ylabel('Im[\Phi]');
title('GHGSF trajectory in complex space');

figure; hold on;
for omega = [0.5, 1.5, 3, 6]
    Phi = (1 / sqrt(2^n * factorial(n) * sqrt(pi) * sigma)) * ...
          hermiteH(n, x) .* exp(-x.^2 / 2) .* ...
          exp(1i * omega * (lambda - mu));
    plot(real(Phi), imag(Phi), 'DisplayName', ['\omega = ', num2str(omega)]);
end
axis equal; grid on;
legend;
title('Effect of modulation frequency in complex space');