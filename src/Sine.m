N = 2e6;
r = linspace(-pi/4, pi/4, N);

sinRef = sin(r);
cosRef = cos(r);

% Sine MinMax Polynomial coff for Homers
c1 =  1.0;
c3 = -1.6666654611e-1;
c5 =  8.3321608736e-3;
c7 = -1.9515295891e-4;
c9 =  2.5925619154e-6;

r2 = r .* r;

sinPoly = r .* (c1 + r2 .* (c3 + r2 .*(c5 + r2 .*(c7 + r2 .*(c9)))));

% Cosine MinMax Polymonail

c2 = -0.5;
c4 =  4.166664568298827e-2;
c6 = -1.388731625493765e-3;
c8 =  2.443315711809948e-5;

cosPoly = c1 + r2 .*(c2 + r2 .*(c4 + r2.*(c6 + r2 .* c8)));

sinErr = sinPoly - sinRef;
cosErr = cosPoly - cosRef;

% Max error
maxSinErr = max(abs(sinErr));
maxCosErr = max(abs(cosErr));

figure;
plot(r, sinErr);
title("Sine MinMax Polynomail Error with ref");
xlabel("r");
ylabel("Error");
grid on;

figure;
plot(r, cosErr);
title("Cos MinMax Polynomail Error with ref");
xlabel("r");
ylabel("Error");
grid on;

