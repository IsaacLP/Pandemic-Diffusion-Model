%% Caso 1
clear; clc; close all

load('reto1_P3.mat')
SIR = [S, I, R];

% Datos de entrenamiento
T = t';
X = [S'; I'; R']./N;

% Suavizar y encontrar r0
[SIR_suavizado,Resultado] = findR0(T,X,3,40,true);
disp(Resultado)

% Resultado
b = Resultado.bmean; g = Resultado.gmean; r0 = Resultado.romean;

% Condiciones inciales para modelo analítico
SIR0 = [5000, 4995, 5, 0];

% Graficar resultados
plotResults(T,SIR,SIR0,SIR_suavizado(:,:,1),b,g,r0,[0 100],1)

%% Caso 2. Net Logo a
clc, clear, close all

% Leer datos de NetLogo
M = readmatrix('SIRNETLOGO5.csv');
S = M(9:end,6); I = M(9:end,2); R = M(9:end,10);
X = [S,I,R];
T = M(9:end,1);
N = 500;
SIR = [S, I, R];

% Datos de entrenamiento
X = X'./N;
T = T';

% Suavizar y encontrar r0
[SIR_suavizado,Resultado] = findR0(T,X,2,1,true);
disp(Resultado)

% Resultado
b = Resultado.bmean; g = Resultado.gmean; r0 = Resultado.romean;

% Condiciones inciales para modelo analítico
SIR0 = [500, 470, 30, 0];

% Graficar resultados
plotResults(T,SIR,SIR0,SIR_suavizado,b,g,r0,[0 192],2)

%% Caso 3 NetLogo b
clc, clear, close all

% Leer datos de NetLogo
M = readmatrix('SIRNETLOGO1B.csv');
S = M(9:end,6); I = M(9:end,2); R = M(9:end,10); N = 500;
T = M(9:end,1);
SIR = [S, I, R]; 

% Datos de entrenamiento
X = [S'; I'; R']./N;
T = T';

% Suavizar y encontrar r0
[SIR_suavizado,Resultado] = findR0(T,X,3,1,true);
disp(Resultado)

% Resultado
b = Resultado.bmean; g = Resultado.gmean; r0 = Resultado.romean;

% Condiciones inciales para modelo analítico
SIR0 = [500, 470, 20, 10];

% Graficar resultados
plotResults(T,SIR,SIR0,SIR_suavizado,b,g,r0,[0 210],3)

%% Caso 4 con moving average
clear; clc; close all;

load('reto4_P3.mat')

% Calcular media movil
SM = movmean(S',200);
IM = movmean(I',200);
RM = movmean(R',200);
SIR = [SM'; IM'; RM'];

% Datos de entrenamiento
T = t;
X = SIR./N;

% Suavizar y encontrar R0
[SIR_suavizado,Resultado] = findR0(T,X,3,10,false);
disp(Resultado)
b = Resultado.bmean; g = Resultado.gmean; r0 = Resultado.romean;

% Condiciones inciales para modelo analítico
SIR0 = [1230, 1220, 10, 0];

% Graficar resultados
plotResults(T,SIR,SIR0,SIR_suavizado(:,:,1),b,g,r0,[0 50],4.1)

%% Caso 4 sin moving avergage
clear; clc; close all

load('reto4_P3.mat')
SIR = [S', I', R'];

% Datos de entrenamiento
T = t;
X = [S; I; R]./N;

% Suavizar y encontrar resultados
tic;
[SIR_suavizado, Resultado] = findR0(T,X,3,10,false);
toc;
disp(Resultado)
b = Resultado.bmean; g = Resultado.gmean; r0 = Resultado.romean;

% Condiciones inciales para modelo analítico
SIR0 = [1275, 1260, 15, 0];

% Graficar resultados
plotResults(T,SIR,SIR0,SIR_suavizado,b,g,r0,[0 50],4.2)

%% Funciones
function [SIR_suavizado,Resultado] = findR0(T,X,neuronas,ciclos,saveall)
% Esta función suaviza los datos con ruido del modelo SIR y encuentra el
% coeficiente R0 con algoritmos genéticos

% Entradas:
% T: Vector de entrada para la red neuronal
% X: Vector de entrenamiento para la red neuronal
% neuronas: Capas de la red nueronal
% ciclos: Número de veces que se repite el algoritmo
% saveall: lógico que indica si se quiere guardar la matriz de
% SIR_suavizado para cada ciclo

% Salidas
% SIR_suavizado: Matriz con el modelo SIR suavizado por la red nueronal
% Resultado: Struct con parámetros obtenidos por un algoritmo genético

% Red neuronal
net = feedforwardnet(neuronas,'trainbr');
net.layers{1}.transferFcn = 'logsig';
net.trainParam.showWindow = false;

% Prealocación
if saveall == true
    SIR_suavizado = zeros(size(X,1),size(X,2),ciclos);
    BG = zeros(ciclos,2);
else
    BG = zeros(ciclos,2);
end

for i = 1:ciclos
    if saveall == true
        % Entrenamiento con moving average
        trainedNet = train(net,T,X);
        SIR_suavizado(:,:,i) = trainedNet(T);
        
        % Encontrar beta y gamma
        TP = diff(T);
        SS = SIR_suavizado(:,:,i);
        SP = diff(SIR_suavizado(1,:,i))./TP;
        RP = diff(SIR_suavizado(3,:,i))./TP;
        
        f = @(w) Error(w,SS,SP,RP);
        
        BG(i,:) = ga(f,2,[],[],[],[],[0 0],[],[],[]);
    else
        % Entrenamiento con moving average
        trainedNet = train(net,T,X);
        SIR_suavizado = trainedNet(T);
        
        % Encontrar beta y gamma
        TP = diff(T);
        SS = SIR_suavizado;
        SP = diff(SIR_suavizado(1,:))./TP;
        RP = diff(SIR_suavizado(3,:))./TP;
        
        f = @(w) Error(w,SS,SP,RP);
        
        BG(i,:) = ga(f,2,[],[],[],[],[0 0],[],[],[]);
    end
end

% Calcular r0
Resultado.beta = BG(:,1);
Resultado.gamma = BG(:,2);
Resultado.ro = BG(:,1) ./ BG(:,2);

% Promedios y desviaciones estándar
Resultado.bmean = mean(Resultado.beta);
Resultado.bdesv = std(Resultado.beta);
Resultado.gmean = mean(Resultado.gamma);
Resultado.gdesv = std(Resultado.gamma);
Resultado.romean = mean(Resultado.ro);
Resultado.rodesv = std(Resultado.ro);
end

function out = Error(w,SS,SP,RP)
% Función para encontrar la beta y gamma que minimicen el error a partir de
% datos de un modelo SIR numérico

b = w(1);
g = w(2);

e1v = zeros(length(SP),1);
e2v = zeros(length(SP),1);

for i=1:length(SP)
    e1v(i) = (SP(1,i) + (b*SS(1,i)*SS(2,i)))^2;
    e2v(i) = (RP(1,i) - (g*SS(2,i)))^2;
end

e1 = sum(e1v);
e2 = sum(e2v);
out = e1 + e2;

end

function plotResults(T,SIR,SIR0,SIR_suavizado,beta,gamma,r0,tspan,caso)
% Función para graficar los resultados finales

% Entradas:
% T: vector de tiempo
% SIR: Modelo SIR numérico de entrada
% SIR0: Condiciones inciales para graficar modelo SIR analítico
% SIR_suavizado: Modelo SIR suavizado por la red neuronal
% beta, gamma, r0: Parámetros encontrados con algoritmo genético
% tspan: Rango de tiempo para graficar el SIR analítico
% caso: Número de caso que se está modelando

% Salidas
% Gráfica que compara los datos de entrada, los datos suavizados, y el
% modelo SIR analítico con los parámetros obtenidos por el algoritmo
% genético

% SIR teórico
N = SIR0(1); S0 = SIR0(2); I0 = SIR0(3); R0 = SIR0(4);

[t_teorico,SIR_teorico] = solveSIR(N,S0,I0,R0,beta,gamma,tspan);

% Gráfica
figure;
P(1:3) = plot(T,SIR./N,'Color',"#0072BD"); hold on;
P(4:6) = plot(T,SIR_suavizado,'Color','k');
P(7:9) = plot(t_teorico,SIR_teorico./N,'Color',"#D95319"); hold on
title(['Caso ' num2str(caso) '. $\beta =$ ' num2str(beta) ...
    ', $\gamma = $ ' num2str(gamma) ...
    ' y $R_0 = $' num2str(r0)] ...
    ,Interpreter='latex')
legend(P([1,4,7]),'Datos de entrada','Datos suavizados','Teórico',Location='best')
xlabel('t')
ylabel('N')
end

function [t,y] = solveSIR(N,S0,I0,R0,beta,gamma,tspan)

odefun = @(t,y) odesir(t,y,N,beta,gamma);
y0 = [S0 I0 R0];
[t,y] = ode45(odefun,tspan,y0);
end

function dydt = odesir(t,y,N,beta,gamma)
dydt = zeros(3,1);
dydt(1) = -beta*y(1)*y(2)/N;
dydt(2) = beta*y(1)*y(2)/N - gamma*y(2);
dydt(3) = gamma*y(2);
end
