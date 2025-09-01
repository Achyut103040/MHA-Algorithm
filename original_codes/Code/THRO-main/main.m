%--------------------------------------------------------------------------
%% %%    Tianji's horse racing optimization (THRO) for 23 functions    %% %%
% THRO code v1.0.                                                          %
%--------------------------------------------------------------------------%                       
% The code is based on the following paper:                                %
% Wang, L., Du, H., Zhang. Z., Hu, G., Mirjalili, S., Khodadadi, N.,       % 
%  Hussien, A.G., Liao, Y., Zhao, W. (2025). Tianji's horse racing         %
% optimization (THRO): A new metaheuristic inspired by ancient wisdom      %
% and its engineering optimization applications, Artificial Intelligence   %
% Review, 58, 282. https://doi.org/10.1007/s10462-025-11269-9.             %
%--------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BestX:The best solution                  %
% BestF:The best fitness                   %
% HisBestF:History of the best fitness     %
% FunIndex:Index of functions              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;

MaxIteration=500;
PopSize=30;

FunIndex=1;
[BestX,BestF,HisBestF]=THRO(FunIndex,MaxIteration,PopSize) ;
display(['The best fitness of F',num2str(FunIndex),' is: ', num2str(BestF)]);
%display(['The best solution is: ', num2str(BestX)]);

if BestF>0
    semilogy(HisBestF,'r','LineWidth',2);
else
    plot(HisBestF,'r','LineWidth',2);
end

xlabel('Iterations');
ylabel('Fitness');
title(['F',num2str(FunIndex)]);



