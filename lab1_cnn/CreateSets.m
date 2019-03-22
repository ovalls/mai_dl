%% 2nd STEP --> 102 Flowers labels create Test and Train sets
% Olga Valls

%% Clear Workspace
clc; close all; clear;
tic

%% Import labels
labels = importdata('imagelabels.mat');

%% Create Train and Test sets
% train: 99% data
% test 1% data. Mínimum amount to appropriately represent each class.

% Quantes hi ha de cada classe:
num_per_class = zeros(1,102);
for i = 1:size(labels,2)
    class = labels(i);
    num_per_class(class) = num_per_class(class) + 1;
end
% sum(num_per_class) --> 8189

% Quantes imatges de cada classe van a test i a train
num_test_per_class = zeros(1,102);
num_train_per_class = zeros(1,102);
for i = 1:size(num_per_class,2) 
    num_test_per_class(i) = floor(0.1 * num_per_class(i));   % 1% imatges per a test
    num_train_per_class(i) = num_per_class(i) - num_test_per_class(i); % la resta per a train
end

% directori on estan les imatges i on aniran les de test i train
folder = 'small_images/';
test_folder = 'test_images/';
train_folder = 'train_images/';

% labels per als datasets de test i de train
labels_test = zeros(1,sum(num_test_per_class));
labels_train = zeros(1,sum(num_train_per_class));

files = dir(fullfile(folder,'*.jpg'));    % pattern to match filenames.
num_files = size(files,1);

acc_test_per_class = zeros(1,102);
% comptadors per a labels_test i labels_train
te = 1;
tr = 1;
for i=1:num_files
    % Read image
    im = imread(fullfile(folder, files(i).name));
    display(files(i).name)   
    % Les primeres de cada classe les posem al test folder.
    % La resta al train folder.
    class = labels(i);
    if (acc_test_per_class(class) < num_test_per_class(class))
        acc_test_per_class(class) = acc_test_per_class(class) + 1;
        labels_test(te) = class;
        % Save image to test_folder
        imwrite(im, fullfile(test_folder, sprintf('test_%05d.jpg', te)));
        te = te+1;
    else
        labels_train(tr) = class;
        % Save image to train_folder
        imwrite(im, fullfile(train_folder, sprintf('train_%05d.jpg', tr)));        
        tr = tr+1;
    end
    
end

%% Name of classes for Keras
% As the 102 classes are from 1 to 102, and we need them as 0-101 for
% Keras, we subtract 1 to each of the values of the labels
for i = 1:size(labels_test,2)
    labels_test(i) = labels_test(i)-1;
end

for i = 1:size(labels_train,2)
    labels_train(i) = labels_train(i)-1;
end


%% Save labels in .csv files
csvwrite('labels_test.csv',labels_test)
csvwrite('labels_train.csv',labels_train)
%type labels_test.csv
%type labels_train.csv

toc
