clear all;
close all;
clc;

load dino_Ps.mat

for i = 1:2
   [K,R,t] = decomposeP(cell2mat(P(i)))
end
    

% dlmwrite('pose.txt', P, 'delimiter','\t','newline','pc')
