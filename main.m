% Make sure to have a fresh start
close all;
clear all;

% Read the localhost port from the config file
parameterFileID = fopen(fullfile('config_files','cam_and_com_config'),'r');
while true
    line = fgetl(parameterFileID);
    if isequal(size(line), [0,0])
        error("Found empty line in config file before finding the localhost.")
    end
    pair = split(line, ', ');
    if pair{1} == "localhost"
        localhost = str2num(pair{2});
        fclose(parameterFileID);
        break
    end
end

% Open figures for UI and create TCP client to communicate with Python
figures = [figure();figure()];
client = tcpclient("localhost",localhost)

% Begin analysis process
write(client,"0","int8"); % This message means that Matlab is waiting for an input image
while true
    % Wait until Python reports that the prediction of the image was saved
    % in disk
    try
        b = read(client, 1);
    catch
        disp("Error receiving data from the remote server. Retrying.")
        b = 0;
    end
    
    if b == 1
        write(client,"2","int8"); % Matlab tells to Python that it is analyzing the image
        disp("Processing new image. Full analysis")
        try
            tic
            inputImg = fullfile('results', 'input_image.tiff'); 
            segmentationPath = fullfile('results', 'interlayer_lines.png');
            analyze(inputImg, segmentationPath, figures); % This is the function used to make all the measurements. The variables as explained in the README file are calculated here.
            toc
        catch
            disp("Analysis error. Skipping frame")
        end
        % If the figure was closed, send a message to Python saying that
        % the tool must stop
        if ~ishandle(figures(1))
            write(client,"-1","int8");
            close all
            break
        end
        write(client,"0","int8"); % Tell Python that MAtlab is waiting for a new image 
    % If the received message was the expected one (1), begin the analysis
    elseif b == 3
        disp("Python could't read a new frame. Exit.")
        close all
        break
    end
end

disp("Process finished.")
clear client % Make sure to clear the client