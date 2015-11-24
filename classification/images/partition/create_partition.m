% CREATE_PARTITION Creates a train/test partition
%
% Usage
%    [train_set, test_set] = CREATE_PARTITION(src, ratio, shuffle)
%
%    [train_set, test_set] = CREATE_PARTITION(obj_class, ratio, shuffle)
%
% Input
%    src (struct): The source structure describing the objects.
%    ratio (numeric, optional): The proportion of all instances selected for 
%       training (default 0.8).
%    shuffle (boolean, optional): If true, objects are shuffled before assign-
%       ing partitions (default 1).
%    obj_class (integer): The indices of the classes each object belongs to.
%       Can be obtained from src through [src.objects.class].
%
% Output
%    train_set (int): The indices of objects in src.objects corresponding to 
%       training instances.
%    test_set (int): The indices of objects in src.objects corresponding to 
%       testing instances.
%
% See also
%    CREATE_SRC, NEXT_FOLD

function [train_set,test_set,valid_set] = create_partition(object_classes, ratio , shuffle)
	
	if nargin < 1
		error('Must specify a source or a list of object classes!');
	end
	
	if nargin < 2
		ratio = 0.8;
	end

	if nargin < 3
		shuffle = 1;
	end

	if isstruct(object_classes)
		src = object_classes;
		object_classes = [src.objects.class];
	end
	
	if length(ratio) == 1
		ratio = [ratio 1-ratio];
	end
	
	if length(ratio) == 2
		ratio = [ratio 0];
                    %  ^ is for the validation set 
	end
	
    if abs( sum(ratio)-1 ) > eps
        error('Ratios must add up to 1!');
    end
	
    number_classes = max( object_classes );
	train_cell = cell( 1 , number_classes ) ;
	test_cell = cell( 1 , number_classes );
	valid_cell = cell( 1 , number_classes );
	
	for class = 1:max( object_classes )
        %% For every class,
		ind = find( object_classes == class);
		
        %% Shuffle
        if shuffle
            ind = ind( randperm(length(ind)) ); 
        end
        
        %% Partition
		train_number = round( ratio(1)*length(ind) );
		if ratio(3) == 0
			test_number = length( ind )-train_number;
			validation_number = 0;
		else
			test_number = round(ratio(2)*length(ind));
			validation_number = length(ind)-train_number-test_number;
		end
		
		train_cell{ class } = ind( 1:train_number );
		test_cell{ class } = ind(train_number+1:train_number+test_number);
		valid_cell{ class } = ind(train_number+test_number+1:train_number+test_number+validation_number);
    end
    train_set = cell2mat( train_cell );
    test_set = cell2mat( test_cell );
    valid_set = cell2mat( valid_cell );

end
