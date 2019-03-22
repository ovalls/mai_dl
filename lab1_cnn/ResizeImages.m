%% 1st STEP --> 102 Flowers resize to 256x256
% Olga Valls

%% Clear Workspace
clc; close all; clear;
tic

%% Resize images to 256x256xchannels
original = 'jpg/';
final = 'small_images/';

files = dir(fullfile(original,'*.jpg'));    % pattern to match filenames.
num_files = size(files,1);

for i=1:num_files
    % Read image
    im = imread(fullfile(original, files(i).name));
    display(files(i).name)

    % Tallo la imatge per a que quedi rectangular
    % [xmin ymin width height]
    % The thing to remember is that the rect is given by [xmin, ymin, width, height]
    % so x comes first, but the matrix is indexed as row, col, so y/row comes first...
    [vert,horiz,c] = size(im);

    if (vert > horiz)
        %display('vert > horiz');
        diff = vert-horiz;
        xmin = 0;
        ymin = floor(diff/2);
        width = horiz;
        height = horiz-1;
    else
        %display('horiz > vert');
        diff = horiz-vert;
        xmin = floor(diff/2);
        ymin = 0;
        width = vert-1;
        height = vert;
    end

    rect = [xmin,ymin,width,height];
    im_crop = imcrop(im,rect);

    % Resize image to 256px x 256px
    im_small = imresize(im_crop,[256, 256]);

    %figure; imshow(im);
    %figure; imshow(im_crop);
    %figure; imshow(im_small);

    % Save resized image to file
    %imwrite(im_small,'small_images/00001.jpg');
    imwrite(im_small, fullfile(final, sprintf('small_%05d.jpg', i))); 
end

toc
