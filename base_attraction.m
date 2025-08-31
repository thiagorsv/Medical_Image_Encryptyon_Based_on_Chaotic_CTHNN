clear all

X = linspace(-6, 6, 301);
Y = linspace(-6, 6, 301);

St = zeros(length(X), length(Y));

tspan = 0:0.01:150;
w23 = 1.2;
for i = 1:length(Y)
    disp(i)
   for j = 1:length(X)
       y0 = [X(j), Y(i), 0];
       [t, y] = ode45(@(t, x) hnn(t, x, w23), tspan, y0);% 0.55<w23< 4
       
       if abs(y(end, 1))>1000
          St(i, j) = 0; % instability
       else
           St(i, j) = 1; % stability
       end
   end
end

%{
% En caso de querer interpolar por razones visuales
[X, Y] = meshgrid(X, Y);
xq = linspace(X(1), X(end), 2*length(X)-1); % 10 veces mas divisiones
yq = linspace(Y(1), Y(end), 2*length(Y)-1); % 10 veces mas divisiones
[Xq, Yq] = meshgrid(xq, yq);
A = interp2(X, Y, St, Xq, Yq, 'linear');
%}
%A = bifu;

figure(2)  
imagesc('XData',X,'YData',Y,'CData',St) 
cmap = parula(2);   % Obtienes el colormap original
colormap(cmap);      % Lo reasignas
cbh = colorbar;
% Activar LaTeX en el colorbar
set(cbh, 'TickLabelInterpreter', 'latex')
xlabel('$x_{0}$', 'Interpreter', 'latex')
ylabel('$y_{0}$', 'Interpreter', 'latex')
set(gca,'fontsize',14)
ax = gca;
ax.TickLabelInterpreter = 'latex';


function y = nonL(u)
    M = 10.5;
    x0 = 0.13333*M;
    if u<-x0
        y = -M;
    elseif u>=-x0 && u<0
        y = -0.3333*M;
    elseif u>=0 && u<x0
        y = 0.3333*M;
    else
        y = 1*M;
    end       
end

function df = hnn(t, v, w23)
    x = v(1); y = v(2); z = v(3);
    
    
    k = 1000;
    w31 = 0.2; w33 = 2.3;
    
    w32 = 10/(w33-0.5*w31);

    w12 = (w32*w23-1.3)/w23;
    
    u = x;
    x0 = 1;
    M = 10.5;
    b = M-x0/k;
    if abs(u)<((b*k)/(k^2+1))
        out = k*u;
    elseif u>=((b*k)/(k^2+1))
        out = u/k +b;
    else
        out = u/k -b;
    end
    psi = out;
    
    W = [0, -w12, 0;
         0, 0,  w23;
        -w31, -w32, w33];
     F = [psi; y; z];
     
     df = -v + W*F;
end
