% CREATE_SRC Create a source of files & objects
%
% Usage
%    src = CREATE_SRC(directory, objects_fun)
%
% Input
%    directory (char): The directory in which the files are found.
%    objects_fun (function handle): Given a filename, objects_fun returns its 
%       constituent objects and their respective classes.
%
% Output
%    src (struct): The source corresponding to all data files (.jpg, .wav, 
%       .au) contained in directory and their objects as defined by 
%       objects_fun.
%
% See also
%    PREPARE_DATABASE

function src = create_src(directory,objects_fun)
	if nargin < 1
		error('Must specify directory!');
	end
	
	if nargin < 2
		objects_fun = @default_objects_fun;
	end
	
	files = find_files(directory);
	
	if isempty(files)
		error('No data files found in the specified directory!');
	end
	
	objects = [];% objects.image : the uint16 image
                 % objects.class : index of its class in classes
	classes = {};% set of the unique classes

    
	for index = 1:length(files)
        
		[file_objects , file_classes] = objects_fun( files{index} );
		
		objects = [objects file_objects];
		classes = [classes file_classes];
    end

	[classes,~,object_class] = unique(classes);
	
	object_class = num2cell(object_class);
	
	[objects.class] = object_class{:};
	
	src.classes = classes;
	src.objects = objects;
end
