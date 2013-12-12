sigmas = [0.5, 1, 2, 3];
x = linspace(-1, 1, 100);

figure(1);
for s = sigmas
    a = -0.5 .* power((x ./ sigma), 2);  

    g = (1/(sigma)*sqrt(2*pi))* exp(a);

    hold on;
    plot(x,g);
    hold off;
end
legend on;