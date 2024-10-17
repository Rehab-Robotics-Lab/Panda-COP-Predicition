classdef ProcessCOP
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Xraw
        Yraw
        Reaction_raw        
        
        X
        Y
        Reaction

        vX
        vY
        
        X_down
        Y_down
        
        X_zero
        Y_zero
        
        rate
        filename

    end
    
    methods
        function obj = ProcessCOP(COPfile,rate)
            obj.rate=rate;
            step=1/rate;
            obj.filename=COPfile;

            % %Checking when the synch light is turned on
            % synch_prompt = "When was the synch light tunred on or reomved from the mat? ";
            % synch_time = input(synch_prompt,'s');
            % synch_time_min = str2double(synch_time(1:2));
            % synch_time_sec = str2double(synch_time(4:5));
            % 
            % %converting input synch light time to frame
            % synch_frame=(synch_time_min*obj.rate*60)+(synch_time_sec*obj.rate);

            % Reading X and Y data without knowing the row the data start
            opts = detectImportOptions(COPfile);
            opts = setvartype(opts,'char');  % or 'string'
            Tbl = readtable(COPfile,opts);

            [a,b]=find(strcmp([Tbl{:,:}], 'X_scaled'));
            [c,d]=find(strcmp([Tbl{:,:}], 'Y_scaled'));
            [e,f]=find(strcmp([Tbl{:,:}], 'Reaction'));

            data=Tbl(a+1:end,b:f);
            datacell=cellfun(@str2num,table2cell(data));
            
            %Reading X and Y
            X=datacell(1:end,1);
            Y=datacell(1:end,2);
            Reaction=datacell(1:end,3);

            obj.Xraw=X;
            obj.Yraw=Y;
            obj.Reaction_raw=Reaction;            

            % %10 window Median Filter on data
            % Xx=medfilt1(X,10);
            % Yy=medfilt1(Y,10);
            % 
            % nX = length(Xx);
            % nY = length(Yy);
            % 
            % % Time vectors based on the sampling rate
            % dtX = step;
            % tX = 0:dtX:(nX - 1) * dtX;
            % 
            % dtY = step;
            % tY = 0:dtY:(nY - 1) * dtY;
            % 
            % % Compute the Fourier Transform
            % fhatX = fft(Xx, nX);
            % fhatY = fft(Yy, nY);
            % 
            % % Create frequency vectors
            % freqX = (0:nX - 1) * (rate / nX);
            % freqY = (0:nY - 1) * (rate / nY);
            % 
            % % Low-pass filter threshold
            % lowPassThreshold = 100; % in Hz
            % aboveThreshold = 15;
            % 
            % % Apply low-pass filter
            % filteredFhatX = fhatX;
            % filteredFhatX(abs(fhatX) < aboveThreshold) = 0;
            % filteredFhatX(abs(freqX) > lowPassThreshold) = 0;
            % 
            % filteredFhatY = fhatY;
            % filteredFhatY(abs(fhatY) < aboveThreshold) = 0;
            % filteredFhatY(abs(freqY) > lowPassThreshold) = 0;
            % 
            % % Inverse Fourier Transform to reconstruct the signals
            % filteredX = real(ifft(filteredFhatX));
            % filteredY = real(ifft(filteredFhatY));
            
            %2nd order Butterworth Filter with 5Hz Cutoff
            freq_cuttoff=5;
            n_order=2;
            [b,a]=butter(n_order,freq_cuttoff/(rate/2));

            obj.X = filtfilt(b,a,X);
            obj.Y = filtfilt(b,a,Y);
            obj.Reaction = filtfilt(b,a,Reaction);

            % plot(obj.Reaction)
            % mean(obj.Reaction)
            
            %Downsampling the data
            obj.X_down=downsample(obj.X,obj.rate);
            obj.Y_down=downsample(obj.Y,obj.rate);
            
            %From old code for zeroing data
            fraction_Time = 0.02; %take 2 seconds of the data as baseline
            bl_timeLim = int16(fraction_Time * length(obj.X)); %number of datapoints to read for baseline
            
            mean_matDataX = mean(obj.X(1:bl_timeLim)); %baseline for x values
            mean_matDataY = mean(obj.Y(1:bl_timeLim)); %baseline for y values
            
            obj.X_zero = obj.X - mean_matDataX; %baseline value subtracted, values averaged around 0
            obj.Y_zero = obj.Y - mean_matDataY;

            %velocity
            obj.vX=diff(obj.X)/step;
            obj.vY=diff(obj.Y)/step;


            
        end
        
        function error_ellipse(obj,condition,size)  
            X=obj.X;
            Y=obj.Y;
            data = transpose([X' ;Y']);            
                       
            covariance = cov(data);
            [eigenvec, eigenval ] = eig(covariance);

            % Get the index of the largest eigenvector
            [largest_eigenvec_ind_c, r] = find(eigenval == max(max(eigenval)));
            largest_eigenvec = eigenvec(:, largest_eigenvec_ind_c);

            % Get the largest eigenvalue
            largest_eigenval = max(max(eigenval));

            % Get the smallest eigenvector and eigenvalue
            if(largest_eigenvec_ind_c == 1)
                smallest_eigenval = max(eigenval(:,2));
                %smallest_eigenvec = eigenvec(:,2);
            else
                smallest_eigenval = max(eigenval(:,1));
                %smallest_eigenvec = eigenvec(1,:);
            end

            % Calculate the angle between the x-axis and the largest eigenvector
            angle = atan2(largest_eigenvec(2), largest_eigenvec(1));

            % This angle is between -pi and pi.
            % Let's shift it such that the angle is between 0 and 2pi
            if(angle < 0)
                angle = angle + 2*pi;
            end

            % Get the coordinates of the data mean
            avg = mean(data);

            % Get the 95% confidence interval error ellipse
            chisquare_val = 2.4477;
            theta_grid = linspace(0,2*pi);
            phi = angle;
            X0=avg(1);
            Y0=avg(2);
            a=chisquare_val*sqrt(largest_eigenval);
            b=chisquare_val*sqrt(smallest_eigenval);

            % area=a*b*pi; 

            % the ellipse in x and y coordinates 
            ellipse_x_r  = a*cos( theta_grid );
            ellipse_y_r  = b*sin( theta_grid );

            %Define a rotation matrix
            R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];

            %let's rotate the ellipse to some angle phi
            r_ellipse = [ellipse_x_r;ellipse_y_r]' * R;

            % Plot the original data
            p2=plot(data(:,1), data(:,2),'.');
            
            p2.Color=[p2.Color 0.1];
            hold on;

            % Draw the error ellipse
            plot(r_ellipse(:,1) + X0,r_ellipse(:,2) + Y0,'-','Linewidth',2)            
            
            % Set the axis labels
            if(size == 'full' )
                axis([-300 300 -300 300])
            elseif (size=='zoom')
                % axis([min(X)-size max(X)+size min(Y)-size max(Y)+size])
                axis([min(X)-20 max(X)+20 min(Y)-20 max(Y)+20])
            end
            
            %Plot name
            if (condition=="none")
                name="No Toy";
            elseif (condition=="feet")
                name="Toy at Feet";
            elseif (condition=="arms")
                name="Toy at Arms";
            else
                name=" ";
            end  
            title("COP Best Fit Ellipse: "+name)
            
            grid on

            xlabel('Xo (mm)');
            ylabel('Yo (mm)');
        end

        % function error_ellipse_full(obj)  
        %     X=obj.X;
        %     Y=obj.Y;
        %     data = transpose([X' ;Y']);            
        % 
        %     covariance = cov(data);
        %     [eigenvec, eigenval ] = eig(covariance);
        % 
        %     % Get the index of the largest eigenvector
        %     [largest_eigenvec_ind_c, r] = find(eigenval == max(max(eigenval)));
        %     largest_eigenvec = eigenvec(:, largest_eigenvec_ind_c);
        % 
        %     % Get the largest eigenvalue
        %     largest_eigenval = max(max(eigenval));
        % 
        %     % Get the smallest eigenvector and eigenvalue
        %     if(largest_eigenvec_ind_c == 1)
        %         smallest_eigenval = max(eigenval(:,2));
        %         %smallest_eigenvec = eigenvec(:,2);
        %     else
        %         smallest_eigenval = max(eigenval(:,1));
        %         %smallest_eigenvec = eigenvec(1,:);
        %     end
        % 
        %     % Calculate the angle between the x-axis and the largest eigenvector
        %     angle = atan2(largest_eigenvec(2), largest_eigenvec(1));
        % 
        %     % This angle is between -pi and pi.
        %     % Let's shift it such that the angle is between 0 and 2pi
        %     if(angle < 0)
        %         angle = angle + 2*pi;
        %     end
        % 
        %     % Get the coordinates of the data mean
        %     avg = mean(data);
        % 
        %     % Get the 95% confidence interval error ellipse
        %     chisquare_val = 2.4477;
        %     theta_grid = linspace(0,2*pi);
        %     phi = angle;
        %     X0=avg(1);
        %     Y0=avg(2);
        %     a=chisquare_val*sqrt(largest_eigenval);
        %     b=chisquare_val*sqrt(smallest_eigenval);
        % 
        %     % the ellipse in x and y coordinates 
        %     ellipse_x_r  = a*cos( theta_grid );
        %     ellipse_y_r  = b*sin( theta_grid );
        % 
        %     %Define a rotation matrix
        %     R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];
        % 
        %     %let's rotate the ellipse to some angle phi
        %     r_ellipse = [ellipse_x_r;ellipse_y_r]' * R;
        % 
        %     % Plot the original data
        %     p2=plot(data(:,1), data(:,2),'.');
        % 
        %     p2.Color=[p2.Color 0.1];
        %     hold on;
        % 
        %     % Draw the error ellipse
        %     plot(r_ellipse(:,1) + X0,r_ellipse(:,2) + Y0,'-','Linewidth',2)
        % 
        %     % Set the axis labels
        %     axis([-300 300 -300 300])
        %     title("COP Best Fit Ellipse")
        %     %axis([-300 300 -300 300])
        % 
        %     grid on
        % 
        %     hXLabel = xlabel('Xo (mm)');
        %     hYLabel = ylabel('Yo (mm)');
        % end
        
        function animate_path(obj)
            X=obj.X;
            Y=obj.Y;
            for i=1:length(X)
               %COP subplot
               cla
               
               p2=plot(X(1:i),Y(1:i),'r');
               p2.Color(4)=0.3;
               hold on
               p1=plot(X(i),Y(i),'b x', 'MarkerSize',15) ;

               axis([min(X)-20 max(X)+20 min(Y)-20 max(Y)+20])
               %axis([-100 100 -100 100])
               grid on
               title('COP')

               xlabel('X(mm)')
               ylabel('Y(mm)')
               
               pause(1/obj.rate)
            end
        end
        
        function animate_path_loop(obj,i)
            X=obj.X_down;
            Y=obj.Y_down;
            
           %COP subplot
           cla

           p2=plot(X(1:i),Y(1:i),'r');
           p2.Color(4)=0.3;
           hold on
           p1=plot(X(i),Y(i),'b x', 'MarkerSize',15) ;

           axis([min(X)-20 max(X)+20 min(Y)-20 max(Y)+20])
           %axis([-100 100 -100 100])
           grid on
           title('COP')

           xlabel('X(mm)')
           ylabel('Y(mm)')
           
           title('Live COP')

           %pause(1/obj.rate)
            
        end      
        
        function plot_XY(obj,deriv,condition) 
            if (deriv==0)
                x=obj.X;
                y=obj.Y;
            elseif (deriv==1)
                x=obj.vX;
                y=obj.vY;
            elseif (deriv==2)
                x=obj.aX;
                y=obj.aY;
            end

            t=0:(length(x)-1);
            t=t/obj.rate;

            hold on

            plot(t,x)           
            plot(t,y)

            if (condition=="none")
                name="No Toy";
            elseif (condition=="feet")
                name="Toy at Feet";
            elseif (condition=="arms")
                name="Toy at Arms";
            else
                name=" ";
            end  
            
            
            grid
            title('Position X and Y plot: '+name)  
            xlabel('time (s)') 
            legend('X','Y')
        end
        
        %Plots to check for rolling
        function plot_rolling(obj,condition,t_offset)  
            if nargin == 2
                t_offset = 0;
            elseif nargin == 1
                condition="ignore";
            end
            %unit mm
            threshold=85;

            X=obj.X;
            frames=0:length(X)-1;
            t=(frames/obj.rate);

            %detecting for anomoly by seeing where the values go less than half of the mean reaction 
            mask_mean=mean(X(1:obj.rate*10));
            % boolean mask for values that deviate from mean by threshold 
            out_mask=abs(X-mask_mean)>=threshold;

            %finidnging how many inatance of rolling there are
            cont=diff(out_mask);
            %appending 0 to the front to fix array size (front because diff works as x(2)-x(1))
            cont=[0;cont];

            %finding frames rolling is detected 
            f_roll_start=frames(cont==1);
            f_roll_end=frames(cont==-1);

            % rolling times that are within 1 seconds of each other and
            % making them contnious
            for i=1:length(f_roll_start)-1
                if f_roll_start(i+1)-f_roll_end(i)<=1*obj.rate
                    out_mask(f_roll_end(i):f_roll_start(i+1))=1;
                end
            end


            %finding times for new coninious distubances
            cont2=diff(out_mask);
            cont2=[0;cont2];
            t_roll_start=t(cont2==1)+t_offset;
            t_roll_end=t(cont2==-1)+t_offset;
            
            %making the end of the roll equal to the max time if the trial ends in rolling 
            if length(t_roll_end)==(length(t_roll_start)-1)
                t_roll_end=[t_roll_end,(max(t)+t_offset)];
            end

            figure(1)
            if (condition=="none")
                name="No Toy";
            elseif (condition=="feet")
                name="Toy at Feet";
            elseif (condition=="arms")
                name="Toy at Arms";
            end 


            %displaying when disturnaces where detected
            disp(name+": Rolling detected "+length(t_roll_start)+" times")
        
            for j=1:length(t_roll_start)
                disp("    Roll "+j+": "+t_roll_start(j)+"s to "+t_roll_end(j)+"s")
            end    

            grid
            
            hold on
            % plot(t(peaklocs),X(peaklocs),'*')
            val=min(650,max(X));
            plot(t,X)
            plot(t,out_mask*val+mask_mean);
            title('Position '+name)
            xlabel('Time')
            ylabel('mm')
            
            %setting azis limits (650 because mat size)
            ylim([max(-650,min(X)), min(650,max(X))])
            xlim([min(t), max(t)])
            
        end
        
        %Plots to check for rolling
        function check_mass(obj,condition,t_offset) 
            if nargin == 2
                t_offset = 0;
            elseif nargin == 1
                condition="ignore";
            end

            r=obj.Reaction;
            frames=0:length(r)-1;
            t=(frames/obj.rate);

            %finding true mass mean by looping through until we've
            %eliminated as much erroneous data
            mask_mean_last=0;
            mask_mean=(mean(r));
            %stopping criteria is finding when difference in current mean mass and previous mean mass is
            %less than 0.2

            while abs(mask_mean-mask_mean_last)>=0.25
                out = r<(mask_mean-0.5);
                mask_mean_last=mask_mean;
                mask_mean=(mean(r(~out)));

            end      
            % % alternative - findinging mask mean as the means reaction from the first 3
            % %seconds
            % mask_mean=mean(r(1:3*obj.rate));

            %portions of the reaction that are less or greater than the
            %mean by the threshold
            thresh=mask_mean*0.25;
            out_mask=abs(r-mask_mean)>=thresh;
            
            %finidnging how many inatance of pick ups there are
            cont=diff(out_mask);
            %appending 0 to the front to fix array size (front because diff works as x(2)-x(1))
            cont=[0;cont];

            %finding frames disturbance is detected 
            f_anom_start=frames(cont==1);
            f_anom_end=frames(cont==-1);

            % disturbance times that are within 5 seconds of each other and
            % making them contnious
            for i=1:length(f_anom_start)-1
                if f_anom_start(i+1)-f_anom_end(i)<=5*obj.rate
                    out_mask(f_anom_end(i):f_anom_start(i+1))=1;
                end
            end


            %finding times for new coninious distubances
            cont2=diff(out_mask);
            cont2=[0;cont2];
            t_anom_start=t(cont2==1)+t_offset;
            t_anom_end=t(cont2==-1)+t_offset;

            %making the end of the roll equal to the max time if the trial ends in rolling 
            if length(t_anom_end)==(length(t_anom_start)-1)
                t_anom_end=[t_anom_end,(max(t)+t_offset)];
            end

            % graph name
            if (condition=="none")
                name="No Toy";
            elseif (condition=="feet")
                name="Toy at Feet";
            elseif (condition=="arms")
                name="Toy at Arms";
            elseif (condition=="ignore")
                name="";
            end 
            
            %displaying when disturnaces where detected
            disp(name+": Anonmilies detected "+length(t_anom_start)+" times")
        
            for j=1:length(t_anom_start)
                disp("    Anomololy "+j+": "+t_anom_start(j)+"s to "+t_anom_end(j)+"s")
            end            
            
            %graph parameters
            hold on
            grid        
        
            title('Check Pickup: '+name)
            xlabel('Time')
            ylabel('mm')
            
            %plotting the reaction
            plot(t,r)
            %plotting dectected mass disturbances
            plot(t,out_mask + mask_mean)
            legend("Mass","Anomolies")

        end

        %Plots to check for rolling
        function anom_bool=check_mass_bool(obj) 

            r=obj.Reaction;
            frames=0:length(r)-1;
            t=(frames/obj.rate);

            %finding true mass mean by looping through until we've
            %eliminated as much erroneous data
            mask_mean_last=0;
            mask_mean=(mean(r));
            %stopping criteria is finding when difference in current mean mass and previous mean mass is
            %less than 0.2

            while abs(mask_mean-mask_mean_last)>=0.25
                out = r<(mask_mean-0.5);
                mask_mean_last=mask_mean;
                mask_mean=(mean(r(~out)));

            end      
            % % alternative - findinging mask mean as the means reaction from the first 3
            % %seconds
            % mask_mean=mean(r(1:3*obj.rate));

            %portions of the reaction that are less or greater than the
            %mean by the threshold
            thresh=mask_mean*0.25;
            out_mask=abs(r-mask_mean)>=thresh;
            
            %finidnging how many inatance of pick ups there are
            cont=diff(out_mask);
            %appending 0 to the front to fix array size (front because diff works as x(2)-x(1))
            cont=[0;cont];

            %finding frames disturbance is detected 
            f_anom_start=frames(cont==1);
            f_anom_end=frames(cont==-1);

            % disturbance times that are within 5 seconds of each other and
            % making them contnious
            for i=1:length(f_anom_start)-1
                if f_anom_start(i+1)-f_anom_end(i)<=5*obj.rate
                    out_mask(f_anom_end(i):f_anom_start(i+1))=1;
                end
            end


            %finding times for new coninious distubances
            cont2=diff(out_mask);
            cont2=[0;cont2];
            t_anom_start=t(cont2==1);
            t_anom_end=t(cont2==-1);

            anom_bool=length(t_anom_start);
        end

        function sections=get_sections(obj) 
            r=obj.Reaction;
            frames=0:length(r)-1;
            t=(frames/obj.rate);

            %finding true mass mean by looping through until we've
            %eliminated as much erroneous data
            mask_mean_last=0;
            mask_mean=(mean(r));
            %stopping criteria is finding when difference in current mean mass and previous mean mass is
            %less than 0.2

            while abs(mask_mean-mask_mean_last)>=0.2
                out = r<(mask_mean-0.5);
                mask_mean_last=mask_mean;
                mask_mean=(mean(r(~out)));

            end      
            % % alternative - findinging mask mean as the means reaction from the first 3
            % %seconds
            % mask_mean=mean(r(1:3*obj.rate));

            %portions of the reaction that are less or greater than the
            %mean by the threshold
            thresh=mask_mean*0.25;
            out_mask=abs(r-mask_mean)>=thresh;
            
            %finidnging how many inatance of pick ups there are
            cont=diff(out_mask);
            %appending 0 to the front to fix array size (front because diff works as x(2)-x(1))
            cont=[0;cont];

            %finding frames disturbance is detected 
            f_anom_start=frames(cont==1);
            f_anom_end=frames(cont==-1);

            % disturbance times that are within 5 seconds of each other and
            % making them contnious
            for i=1:length(f_anom_start)-1
                if f_anom_start(i+1)-f_anom_end(i)<=5*obj.rate
                    out_mask(f_anom_end(i):f_anom_start(i+1))=1;
                end
            end


            %finding times for new coninious distubances
            cont2=diff(out_mask);
            cont2=[0;cont2];
            f_anom_start=frames(cont2==1);
            f_anom_end=frames(cont2==-1);

            %making the end of the roll equal to the max time if the trial ends in rolling 
            if length(f_anom_end)==(length(f_anom_start)-1)
                f_anom_end=[f_anom_end,max(frames)];
            end
            
            %making an array with the starting and stopping time of the
            %different sections
            n=length(f_anom_end);
            if n>0
                section=[1,f_anom_start(1)];
    
                for i=1:(length(f_anom_end)-2)
                    section=[section;[f_anom_end(i),f_anom_start(i+1)]];
                end
    
                section=[section;[f_anom_end(end),max(frames)]];
            else
                section=[1,max(frames)];
            end

            % deleting last row if last section is only one row (happens if session ends in anomoly)
            if section(end,1)==section(end,2)
                section(end,:)=[];
            end
            sections=section;  
        end

        function plot_sections(obj,condition) 
            % graph name
            if (condition=="none")
                name="No Toy";
            elseif (condition=="feet")
                name="Toy at Feet";
            elseif (condition=="arms")
                name="Toy at Arms";
            elseif (condition=="ignore")
                name="";
            end 
            
            %getting the section locations
            sections=get_sections(obj);

            % graph parameters
            hold on
            grid        

            title('Relevant Section: '+name)
            xlabel('Time')
            ylabel('mm')
            
            %plotting the COP
            t=0:(length(obj.X)-1);
            t=t/obj.rate;

            hold on

            plot(t,obj.X)           
            plot(t,obj.Y)

            [row,col]=size(sections);
            
            %plotting sections of COP we'll be using
            for i=1:row
                plot(t(sections(i,1):sections(i,2)),obj.X(sections(i,1):sections(i,2)),color=[0.9290 0.6940 0.1250]);
                plot(t(sections(i,1):sections(i,2)),obj.Y(sections(i,1):sections(i,2)),color=[0.9290 0.6940 0.1250]);
            end           
            % plot(t(section_mask),obj.Y(section_mask),'green') 

            legend("X","Y","Sections")
            hold off

        end

        %calculating STD of the position
        function [stdX,stdY]=std(obj)
           stdX=std(obj.X_zero);
           stdY=std(obj.Y_zero);
        end

        %calculating STD of the velocity
        function [stdX,stdY]=std_V(obj)
           stdX=std(obj.vX);
           stdY=std(obj.vY);
        end
        
        %calculating RMS of the position
        function rmsCOP=RMS(obj)
           curX = abs(obj.X_zero); %renamed as to not tamper with existing math below
           curY = abs(obj.Y_zero);
            
           copMag = sqrt(curX.^2 + curY.^2);

           rmsCOP=rms(copMag);
        end

        %calculating RMS of the position
        function rmsCOP=RMS_V(obj)
           curX = abs(obj.vX); %renamed as to not tamper with existing math below
           curY = abs(obj.vY);
            
           copMag = sqrt(curX.^2 + curY.^2);

           rmsCOP=rms(copMag);
        end
        
        %calculating average path length
        function avg_pathLen=path_length(obj)          
            %Rectify data 
            curX = abs(obj.X_zero); %renamed as to not tamper with existing math below
            curY = abs(obj.Y_zero);
            
            copMag = sqrt(curX.^2 + curY.^2);
            pathLen = zeros(size(copMag));

            for i = 2:1:size(pathLen) - 1
                pathLen(i) =  sqrt((curX(i) - curX(i - 1)).^2 + (curY(i) - curY(i - 1)).^2);
            end

            avg_pathLen=sum(pathLen)/(size(copMag,1));           
        end
        
        %calculating ellipse area
        function area=ellipse_area(obj)
            X=obj.X';
            Y=obj.Y';
            
            data = transpose([X ;Y]); 
            % Calculate the eigenvectors and eigenvalues
            covariance = cov(data);
            [eigenvec, eigenval ] = eig(covariance);

            % Get the index of the largest eigenvector
            [largest_eigenvec_ind_c, r] = find(eigenval == max(max(eigenval)));

            % Get the largest eigenvalue
            largest_eigenval = max(max(eigenval));

            % Get the smallest eigenvector and eigenvalue
            if(largest_eigenvec_ind_c == 1)
                smallest_eigenval = max(eigenval(:,2));
            else
                smallest_eigenval = max(eigenval(:,1));
            end

            % Get the 95% confidence interval error ellipse
            chisquare_val = 2.4477;
            a=chisquare_val*sqrt(largest_eigenval);
            b=chisquare_val*sqrt(smallest_eigenval);
            
            area=a*b*pi;            
        end 
        
        %calculating excursion of the position
        function [excursionX,excursionY]=excursion(obj)
           excursionX=max(obj.X)-min(obj.X);
           excursionY=max(obj.Y)-min(obj.Y);
        end

        %calculating excursion of the velocity
        function [excursionX,excursionY]=excursion_V(obj)
           excursionX=max(obj.vX)-min(obj.vX);
           excursionY=max(obj.vY)-min(obj.vY);
        end
        
        %calculating SPECIFIC entropy of the position
        function [entX,entY]=entropy(obj)
           % entX=approximateEntropy(obj.X);
           % entY=approximateEntropy(obj.Y);
            detrend_Xraw = (obj.Xraw-mean(obj.Xraw,1))/std(obj.Xraw);
            detrend_Yraw = (obj.Yraw-mean(obj.Yraw,1))/std(obj.Yraw);

            dim=2;

            entX=SampEn(dim, 0.2, detrend_Xraw);
            entY=SampEn(dim, 0.2, detrend_Yraw);

            % Nraw = length(obj.Xraw);

        end

        %calculating SPECIFIC entropy of the velocity
        function [ventX,ventY]=entropy_V(obj)
           % entX=approximateEntropy(obj.vX);
           % entY=approximateEntropy(obj.vY);
           detrend_vX = (obj.vX-mean(obj.vX,1))/std(obj.vX);
           detrend_vY = (obj.vY-mean(obj.vY,1))/std(obj.vY);

           dim=2;

           ventX=SampEn(dim, 0.2, detrend_vX);
           ventY=SampEn(dim, 0.2, detrend_vY);
        end
        
        %calculating mean of the position
        function [meanX,meanY]=mean(obj)
           meanX=mean(obj.X_zero,1);
           meanY=mean(obj.Y_zero,1);
        end

        %calculating mean of the velocity
        function [meanX,meanY]=mean_V(obj)
           meanX=mean(obj.vX,1);
           meanY=mean(obj.vY,1);
        end
        
        %calculating median of the position
        function [medX,medY]=median(obj)
           medX=median(obj.X_zero,1);
           medY=median(obj.Y_zero,1);
        end

        %calculating median of the velocity
        function [medX,medY]=median_V(obj)
           medX=median(obj.vX,1);
           medY=median(obj.vY,1);
        end

        %calculating time spent rolling
        function t=rolling(obj)
           % [peaks,abspeaks,widths,prom]=findpeaks(abs(obj.X),'MinPeakProminence',85,'WidthReference','halfheight');
           % t=sum(widths)/length(obj.X);
           if nargin == 2
                t_offset = 0;
            elseif nargin == 1
                condition="ignore";
            end
            %unit mm
            threshold=85;

            X=obj.X;
            frames=0:length(X)-1;
            t=(frames/obj.rate);

            %detecting for anomoly by seeing where the values go less than half of the mean reaction 
            %in case the section is shorter than 10 seconds
            if length(X)<(obj.rate*10)
                mask_mean=mean(X);
            else
                mask_mean=mean(X(1:obj.rate*10));
            end
            % boolean mask for values that deviate from mean by threshold 
            out_mask=abs(X-mask_mean)>=threshold;

            %finidnging how many inatance of rolling there are
            cont=diff(out_mask);
            %appending 0 to the front to fix array size (front because diff works as x(2)-x(1))
            cont=[0;cont];

            %finding frames rolling is detected 
            f_roll_start=frames(cont==1);
            f_roll_end=frames(cont==-1);

            % rolling times that are within 1 seconds of each other and
            % making them contnious
            for i=1:length(f_roll_start)-1
                if f_roll_start(i+1)-f_roll_end(i)<=1*obj.rate
                    out_mask(f_roll_end(i):f_roll_start(i+1))=1;
                end
            end
            % finidnind the normalized rolling time 
            t=sum(out_mask)/length(out_mask);
             
        end
        
        %Outpputing row of all metricssections=fix_mass(obj,condition,t_offset
        function table=metrics(obj)

           [stdX,stdY]=std(obj);
           [vstdX,vstdY]=std_V(obj);

           rms=RMS(obj);
           vrms=RMS_V(obj);

           area=ellipse_area(obj);

           [extX,extY]=excursion(obj);
           [vextX,vextY]=excursion_V(obj);

           pathLen=path_length(obj);

           [entX,entY]=entropy(obj);
           [ventX,ventY]=entropy_V(obj);

           [meanX,meanY]=mean(obj);
           [vmeanX,vmeanY]=mean_V(obj);

           [medX,medY]=median(obj);
           % [vmedX,vmedY]=median_V(obj);

           troll=rolling(obj);

           row=[stdX,stdY,rms,pathLen,extX,extY,entX,entY,area,...
               vstdX,vstdY,vrms,vextX,vextY,ventX,ventY,vmeanX,vmeanY,troll,meanX,meanY,medX,medY];


           table=array2table(row,'VariableNames', ...
       {'COP Std X','COP Std Y','COP RMS','COP Path Length', ...
       'COP excursion X','COP excursion Y','COP entropy X','COP entropy Y', ...
       'COP area','COP Std vX','COP Std vY','COP vRMS','COP excursion vX','COP excursion vY',...
       'COP entropy vX','COP entropy vY','COP mean vX','COP mean vY','Time Rolling',...
       'COPMeanX','COPMeanY','COPMedianX', 'COPMedianY'...
       });
        end
        
        %Getting metrics for data where only sections can be looked at
        function table=metrics_sections(obj) 
            sections=get_sections(obj);

            X_og=obj.X;
            Y_og=obj.Y;
            X_zero_og=obj.X_zero;
            Y_zero_og=obj.Y_zero;
            vX_og=obj.vX;
            vY_og=obj.vY;

            obj.X=X_og(sections(1,1):sections(1,2));
            obj.Y=Y_og(sections(1,1):sections(1,2));
            obj.X_zero=X_zero_og(sections(1,1):sections(1,2));
            obj.Y_zero=Y_zero_og(sections(1,1):sections(1,2));
            obj.vX=vX_og(sections(1,1):sections(1,2));
            obj.vY=vY_og(sections(1,1):sections(1,2));

            tables=metrics(obj);

            %middle sections
            for i=2:(length(sections)-1)
                obj.X=X_og(sections(i,1):sections(i,2));
                obj.Y=Y_og(sections(i,1):sections(i,2));
                obj.X_zero=X_zero_og(sections(i,1):sections(i,2));
                obj.Y_zero=Y_zero_og(sections(i,1):sections(i,2));
                obj.vX=vX_og(sections(i,1):sections(i,2));
                obj.vY=vY_og(sections(i,1):sections(i,2));

                tables=[tables;metrics(obj)];
            end
            
            %last section
            obj.X=X_og(sections(end,1):sections(end,2));
            obj.Y=Y_og(sections(end,1):sections(end,2));
            obj.X_zero=X_zero_og(sections(end,1):sections(end,2));
            obj.Y_zero=Y_zero_og(sections(end,1):sections(end,2));
            obj.vX=vX_og(sections(end,1):length(vX_og));
            obj.vY=vY_og(sections(end,1):length(vY_og));

            tables=[tables;metrics(obj)];
            table=mean(tables,1);

            obj.X=X_og;
            obj.Y=Y_og;
            obj.X_zero=X_zero_og;
            obj.Y_zero=Y_zero_og;
            obj.vX=vX_og;
            obj.vY=vY_og;
         
        end
        
        function table=empty_metrics(obj)
            row=nan(1,23);     
       
           table=array2table(row,'VariableNames', ...
       {'COP Std X','COP Std Y','COP RMS','COP Path Length', ...
       'COP excursion X','COP excursion Y','COP entropy X','COP entropy Y', ...
       'COP area','COP Std vX','COP Std vY','COP vRMS','COP excursion vX','COP excursion vY',...
       'COP entropy vX','COP entropy vY','COP mean vX','COP mean vY','Time Rolling', ...
       'COPMeanX','COPMeanY','COPMedianX', 'COPMedianY'...
       });
        end
        
            
    end
    
end

