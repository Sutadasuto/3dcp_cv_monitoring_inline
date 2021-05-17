close all;
clear all;

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

f = figure();
client = tcpclient("localhost",localhost)

write(client,"0","int8");
while true
    try
        b = read(client, 1);
    catch
        disp("Error receiving data from the remote server. Retrying.")
        b = 0;
    end
    if b == 1
        write(client,"2","int8");
        disp("Processing new image. Full analysis")
        try
            inputImg = fullfile('tmp', 'img.jpg'); 
            segmentationPath = strrep(inputImg,'.jpg','_gt.png');
            analyze(inputImg, segmentationPath, f);
        catch
            disp("Analysis error. Skipping frame")
        end
        if ~ishandle(f)
            write(client,"-1","int8");
            close all
            break
        end
        write(client,"0","int8");
    elseif b == 3
        disp("Python could't read a new frame. Exit.")
        close all
        break
    end
end

disp("Process finished.")
clear client