function[] = proj2(realData,syntheticData)

filename = realData;
data = importdata(filename);
len = length(data);
 regr_input = zeros(len,47);
 vector = zeros(1,47);

for i=1:len
    single =data{i};
    rel = str2double(single(1));
    occurence = strfind(single,':');
    o = length(occurence);
    vector(1) = rel;
    for j=2:o
        vector(j) = str2double(single(occurence(j)+1:occurence(j)+8));
    end
    
    regr_input(i,:) = vector;
end

train = regr_input(1:55698,:);
valid = regr_input(55699:62660,:);
test = regr_input(62661:69623,:);

% save('train_data.mat','train')
% save('valid_data.mat','valid')
% save('test_data.mat','test')

%train = load('train_data.mat');
target = train(:,1);
input = train(:,2:47);

E = zeros(46,46);


for i = 1:46
    E(i,i) = var(input(:,i))/50;
end
[Wml,DesMat,Sigma1] = train_batch(input,target,E);


Vout = valid(:,1);
Vin = valid(:,2:47);
[validPer1] = validate_realBatch(Vin,Vout,Sigma1,Wml);


[wgrad,w_delta] = train_realGrad(target,DesMat);



data = load(syntheticData);
inputs = data.x';
targets = data.t;

Es = zeros(10,10);


for i = 1:10
    Es(i,i) = var(inputs(:,i))/50;
 end

[Wmls,DesMats] = synthetic_train(inputs,targets,Es);
[validPer2] = validate_syntheticBatch(DesMats,Wmls,targets);

[wgrad_syn,wDelta_syn]= train_synGrad(targets,DesMats);

end

%%%%%%%%%%%%%%%%%ending of main function

function [Wml,DesMat,Sigma1] = train_batch(x,y,E)

mea = [1 2 3 4 5 6 7 8 9 10];
mea = mea/10;
lambda1 = 4000;
M1 = 11;
[lenx,~] = size(x);
lenm = length(mea);

mu1 =  ones(46,M1);
Sigma1 = zeros(46,46,M1);
for k=1:46
    
    mu1(k,:) = [1 mea];
    
end

for g=1:M1
    Sigma1(:,:,g)= E + eye(length(E))*1e-3;
end


DesMat = ones(lenx,M1);
for i=1:lenx
    for j=1:lenm
        DesMat(i,j+1)= exp(-((x(i,:)- mea(j))/Sigma1(:,:,j+1)*(x(i,:)- mea(j))')/2);
    end
end

Wml = pinv(lambda1*eye(length(DesMat'*DesMat)) + (DesMat'*DesMat))* DesMat'* y;
w1 = Wml;
%
%save('hyper_parameters.mat','mean','lambda','M','E')

%
trainInd1 = (1:55698)';
validInd1 = (55699:62660)';
error = (Wml'*DesMat') - y';

vect=error*error';
meanErr=vect/length(y);
trainPer1=sqrt(meanErr);

%fprintf('the trainig error using batch on real data %4.2f\n',trainPer1);


save('proj2.mat','M1','mu1','Sigma1','lambda1','trainInd1','validInd1','w1','trainPer1');
end


%%%%%%

function [ validPer1 ] = validate_realBatch(x,y,S,Wml )
mea = [1 2 3 4 5 6 7 8 9 10];
mea = mea/10;
lambda1 = 3;
M1 = 11;
[lenx,~] = size(x);
lenm = length(mea);

DesMat = ones(lenx,M1);
for i=1:lenx
    for j=1:lenm
        DesMat(i,j+1)= exp(-((x(i,:)- mea(j))/S(:,:,j+1)*(x(i,:)- mea(j))')/2);
    end
end

error = (Wml'*DesMat') - y';

vect=error*error';
meanErr=vect/length(y);
validPer1=sqrt(meanErr);

save('proj2.mat','validPer1','-append');

%fprintf('the validation error on real data using batch is %4.2f\n',validPer1);


end

%%%%%%%%%%
function [ wgrad,w_delta ] = train_realGrad(y,D)
n1 = 0.02;
l = load('proj2.mat','lambda1');
lam = l.lambda1;
tempArr = zeros(55698,1);
w_delta = zeros(11,55698);
wgrad = zeros(11,1);
w01 = wgrad;
eta1 = zeros(1,55698);
for i = 1:55698
eta1(:,i) = n1;
update = (n1*transpose(D(i,:))* (y(i,:)- (transpose(wgrad)*transpose(D(i,:)))));% - (n1*lam*wgrad);
w_delta(:,i) = update;
wgrad = wgrad + update;
predictHyp = wgrad'*D';
err = predictHyp-y';
vet = err*err';
currErr = sqrt(vet/length(y));
tempArr(i) = currErr;
end


dw1 = w_delta;


%fprintf('error on real data during SGD is %4.2f\n',currErr);


save('proj2.mat','eta1','w01','dw1','-append');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ Wmls,DesMats ] = synthetic_train(xs,ys,Es)

meas = [1 2 3 4 5 6];
meas = meas/6;
lambda2 = 0.04;
M2 = 7;
[lenx,~] = size(xs);
lenm = length(meas);

mu2 =  ones(10,M2);
Sigma2 = zeros(10,10,M2);
for k=1:10
    
    mu2(k,:) = [1 meas];
    
end

for g=1:M2
    Sigma2(:,:,g)= Es + eye(length(Es))*1e-3;
end

%tt = inv(Sigma2(:,:,1));
%c = cond(Sigma2(:,:,1));


DesMats = ones(lenx,M2);
for i=1:lenx
    for j=1:lenm
        DesMats(i,j+1)= exp(-((xs(i,:)- meas(j))/Sigma2(:,:,j+1)*(xs(i,:)- meas(j))')/2);
    end
end

D = DesMats(1:1600,:);
Y = ys(1:1600,:);
Wmls = pinv(lambda2*eye(length(D'*D)) + (D'*D))* D'* Y;
w2 = Wmls;
%
%save('hyper_parameters.mat','mean','lambda','M','E')

%
trainInd2 = (1:1600)';
validInd2 = (1601:2000)';
error = (w2'*D') - Y';

vect=error*error';
meanErr=vect/length(Y);
trainPer2=sqrt(meanErr);
%fprintf('error during trainig on syn data is %4.2f\n',trainPer2);


save('proj2.mat','M2','mu2','Sigma2','lambda2','trainInd2','validInd2','w2','trainPer2','-append');
end



%%%%%%%%%%%%%%%%%%%%%%%

function [validPer2] = validate_syntheticBatch(D,W,t)

Des = D(1601:2000,:);
tar = t(1601:2000,:);

error = (W'*Des') - tar';

vect=error*error';
meanErr=vect/length(tar);
validPer2=sqrt(meanErr);
save('proj2.mat','validPer2','-append');
%fprintf('error during validation on syn data is %4.2f\n',validPer2);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [wGrad,wdelta ] = train_synGrad(t,De)
y = t(1:1600,:);
D = De(1:1600,:);
l = load('proj2.mat','lambda2');
lam = l.lambda2;
n2 = 0.01;
wdelta = zeros(7,1600);
wGrad = zeros(7,1);
w02 = wGrad;
eta2 = zeros(1,1600);
for i = 1:1600
eta2(:,i)= n2;
update = (n2*transpose(D(i,:))* (y(i,:)- (transpose(wGrad)*transpose(D(i,:)))));% -  (n2*lam*wGrad);
wdelta(:,i) = update;
wGrad = wGrad + update;
predictHyp = wGrad'*D';
err = predictHyp-y';
vet = err*err';
currErr = sqrt(vet/length(y));

end
fprintf('error on syn data during SGD is %4.2f\n',currErr);


dw2 = wdelta;

save('proj2.mat','eta2','w02','dw2','-append');




end






