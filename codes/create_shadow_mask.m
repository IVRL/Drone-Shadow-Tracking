frame_folder = './data_frames/VIDEO_NAME_frames'; % please change VIDEO_NAME
masks_folder = './VIDEO_NAME_mask/'; % please change VIDEO_NAME

if(~exist(masks_folder, 'dir'))
    disp(['Creating directory ' masks_folder]);
    mkdir(masks_folder);
end

filenames = dir(fullfile(['./' frame_folder '*.jpg']));  % load all frames
n = length(filenames);  % total number od frames
sensitivity2 = 0.425;

for i=1:length(filenames)

    img_name = filenames(i).name;
    disp(['Processing image ' img_name ' (' num2str(i) '/' num2str(n)  ')...']);
    I = imread([frame_folder img_name]);

    imgray = rgb2gray(I); 
    nbhood = 2*floor(size(imgray)/16)+1; 
    T = adaptthresh(imgray,sensitivity2,'NeighborhoodSize',nbhood,'ForegroundPolarity','dark');
    msk = imbinarize(imgray,T);  % >T =1(white)   <T = 0(black)
    imwrite(msk, [masks_folder, img_name,'_',num2str(sensitivity2),'.jpg']); 

end

disp("--- All done ---");