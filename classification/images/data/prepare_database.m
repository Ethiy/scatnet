% PREPARE_DATABASE Calculates the features from objects in a source
%
% Usage
%    database = PREPARE_DATABASE(src, feature_fun, options)
%
% Input
%    src (struct): The source specifying the objects.
%    feature_fun (cell): The feature functions applied to each object.
%    options (struct): Options for calculating the features:
%       options.feature_sampling (int): specifies how to sample the feature 
%           vectors in time/space (default 1).
%       options.file_normalize (int): The normalization of each file before 
%          being given to feature_fun. Can be empty, 1, 2, or Inf (default []).
%       options.parallel (boolean): If true, tries to use the Distributed 
%          Computing Toolbox to speed up calculation (default true).
%       Other options are listed in the help for the FEATURE_WRAPPER function.
% Output
%    database (struct): The database of feature vectors.
%
% See also
%    CREATE_SRC, FEATURE_WRAPPER

function db = prepare_database( src , feature_fun , opt )
    if nargin < 3
        opt = struct();
    end
    
    opt = fill_struct(opt, 'feature_sampling', 1);
    opt = fill_struct(opt, 'file_normalize', []);
    opt = fill_struct(opt, 'parallel', 1);

    features = [];
    
    if opt.parallel
        % parfor loop - note that the contents are the same as the serial loop, but
        % MATLAB doesn't seem to offer an easy way of deduplicating the code.

        % Slice variables for parfor
        objects = src.objects( : );
        normalize = opt.file_normalize;
        sampling = opt.feature_sampling;

        %% Loop through all the files in the source

        parfor iterator = 1:length( src.objects )
            tic
            %% Load the complete file and normalize as needed.
            x = double( objects( iterator ).image );

            %% Normalize
            if ~isempty( normalize )
                switch normalize
                    case 1
                        x = x/sum(abs(x(:)));
                    case 2
                        x = x/sqrt(sum(abs(x(:)).^2));
                    case Inf
                        x = x/max(abs(x(:)));
                    otherwise
                        error('Unknown norm');
                end
             end

            %% Apply all the feature functions to the objects contained in x. 

            buf = feature_fun( x );

            % Subsample among the features as needed.
            features = [ features ; buf( : , 1:sampling:end )];

            fprintf('.');
        end
    else
        fprintf('[] 00%%');
        for iterator = 1:length( src.objects )
            
            %% Load the complete file and normalize as needed.
            x = double( src.objects( iterator ).image );

            %% Normalize

            if ~isempty( opt.file_normalize )
                switch opt.file_normalize
                    case 1
                        x = x/sum(abs(x(:)));
                    case 2
                        x = x/sqrt(sum(abs(x(:)).^2));
                    case Inf
                        x = x/max(abs(x(:)));
                    otherwise
                        error('Unknown norm');
                end
            end

            %% Apply all the feature functions to the objects contained in x.

            buf = feature_fun( x );

            %% Subsample among the features as needed.
            features = [ features ; buf( : , 1:opt.feature_sampling:end  )];
            
            fprintf('\b\b\b\b\b.] %2.0f%%' , iterator/length(src.objects ) * 100);
        end
        fprintf(' \n');
    end
	
    db.src = src;
    db.features = features;
    
end