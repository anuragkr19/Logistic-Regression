%external reference used
% https://www.coursera.org/learn/machine-learning/home/week/3
% https://www.coursera.org/learn/machine-learning/programming/ixFof/logistic-regression

%Main entry of the program
function   main()
% Load Data
T = readtable('inputdata.txt','Format','%f%f%f%s');
Main_table = T(:,1:4);

X = T(:,1:3);
%Converting Training data to array format
X = table2array(X);
y = T(:,4);

%converting Training label to array format
yt = table2array(y);

%creating 1d arary with zeros value
ytrain = zeros(length(yt),1);

N = length(ytrain);

%loading test data
TestTable = readtable('sampledata.txt','Format','%f%f%f');
XTest = TestTable(:,1:3);
XTest = table2array(XTest);

%end loading  data

%preprocessing data

%converting training labels to 1 and 0 and stroing it in the variable
%ytrain

for i=1:length(yt)
    j = char(T{i,4});
    if(j == 'W')
        ytrain(i,:) = 0;
    else
        ytrain(i,:) = 1;
    end
end



%Data Preprocessing

% Preparing data matrix using Training dataset
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize the fitting parameters
initial_theta = zeros(n+1, 1);

% Add intercept term to XTest

XTest = [ones(length(XTest), 1) XTest];

%converted output label 0 represents W and 1 represent M respectively
YTest = [0,1,0,1];

%end Data Preprocessing


%compute parameters theta using gradientdescent algorithm
alpha = 0.01;
noofiterations = 1200;
[theta] = gradientDescent(X, ytrain, initial_theta, alpha, noofiterations);



%  Predict probability for a training data set
for k = 1:length(XTest)
    prob = sigmoid(XTest(k,:) * theta);
    fprintf('For a test data %d %d %d %d, predicted class label is %d and true class label was %d\n', XTest(k,:),round(prob),YTest(k));
end

%end of programe

end

