load 'predictions_prej_fine.mat'

clc;

model = 1;

preds = R(model,:);

b = reshape(preds, 5, 720/5);

nobjects = 5;
depths = 6;
orients = 4;
nviews = 3;
ninstances = 2;

%% Global analysis

fprintf('Global results\n');
fprintf('==============\n');

acc = sum(gt == preds) / length(preds)

confusionmat(gt,preds)

%% Depth level local analyses

fprintf('Partial results\n');
fprintf('===============\n');
accLevels = zeros(1,6);

for d = 1:6
    
    x = linspace(1,nobjects,nobjects) - 1;
    xx = repmat(x, depths*orients*nviews*ninstances, 1);
    gt = reshape(xx, prod(size(xx)), 1)';

    %base = [(tdepths(1)-1)*orients+1, (tdepths(2))*orients ];
    base = [(d-1)*orients+1, (d)*orients ];

    lgt = []; 
    lpreds = [];    

    for o = 1:nobjects
        for i = 1:nviews
            for j = 1:ninstances

                indices = base + ...
                    ((o-1) * nviews * ninstances * depths * orients) + ...
                    ((i-1) * ninstances * depths * orients) + ...
                    ((j-1) * depths * orients);

                lgt = [lgt, gt(indices(1):indices(2))];
                lpreds = [lpreds, preds(indices(1):indices(2))];
            end
        end
    end
    
    fprintf('d = %i\n', d);
    
    accLevel = sum(lgt == lpreds)/length(lgt);
    accLevels(d) = accLevel;
    
    confusionmat(lgt, lpreds)
end

fprintf('Summary of partial analyses\n');

accLevels