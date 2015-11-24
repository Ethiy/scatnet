% FIND_FILES lists interesting files in the given directiry
%
% Usage
%    files = find_files(directory)
%
% Input
%    directory (char): The directory in which the files are found.
%    
%
% Output
%    files : the files list
%
% See also
%    CREATE_SRC

function files = find_files(directory)
	extensions = {'au','wav','jpg','png','tif'};
	
	directory_list = dir(directory);
	
	files = {};
	
	for file_iterator = 1:length( directory_list )
		file_name = directory_list( file_iterator ).name;
		
		% Skip hidden file or current/upper directory.
		if file_name(1) == '.'
			continue;
		end
		
		% Depending on file type, recurse or add audio/image file.
		if directory_list(file_iterator).isdir
			files = [files find_files( fullfile( directory , file_name ) ) ];
		else
			type_found = 0;
			for file_type = 1:length(extensions)
				extention_length = length(extensions{file_type});
				if length(file_name) > extention_length+1 && ...
				   strcmpi(file_name(end-extention_length:end),['.' extensions{file_type}])
					type_found = file_type;
					break;
				end
			end
			if type_found > 0
				files = [files fullfile(directory,file_name)];
			end
		end
	end
end
