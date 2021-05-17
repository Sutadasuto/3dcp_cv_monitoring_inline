% Make sure to have a fresh start
close all;
clear all;

% Read the localhost port from the config file
parameterFileID = fopen('config','r');
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

% Open figure for UI and create TCP client to communicate with Python
f = figure();
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
    % If the received message was the expected one (1), begin the analysis
    if b == 1
        write(client,"2","int8"); % Matlab tells to Python that it is analyzing the image
        disp("Processing new image. Full analysis")
        try
            inputImg = fullfile('tmp', 'img.jpg'); 
            segmentationPath = strrep(inputImg,'.jpg','_gt.png');
            analyze(inputImg, segmentationPath, f); % This is the function used to make all the measurements. The variables as explained in the README file are calculated here.
        catch
            disp("Analysis error. Skipping frame")
        end
        % If the figure was closed, send a message to Python saying that
        % the tool must stop
        if ~ishandle(f)
            write(client,"-1","int8");
            close all
            break
        end
        write(client,"0","int8"); % Tell Python that MAtlab is waiting for a new image
    elseif b == 3
        disp("Python could't read a new frame. Exit.")
        close all
        break
    end
end

disp("Process finished.")
clear client % Make sure to clear the client