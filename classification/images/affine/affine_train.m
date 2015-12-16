% AFFINE_TRAIN Train an affine space classifier
%
% Usage
%    model = AFFINE_TRAIN(db, train_set, options)
%
% Input
%    db (struct): The database containing the feature vector.
%    train_set (int): The object indices of the training instances.
%    options (struct): The training options. options.dim specifies the dimen-
%        sionality of the affine spaces modeling each class.
%
% Output
%    model: The affine space model.
%
% See also
%    AFFINE_TEST, AFFINE_PARAM_SEARCH

function model = affine_train(db,train_set,opt)
	if nargin < 3
		opt = struct();
	end
	
	%% Set default options.
	opt = fill_struct( opt , 'dim' , 80 );
	
	%% Create mask to separate the training vectors
	train_mask = ismember(1:length(db.src.objects),train_set);
	
    mu = cell( 1 , length(db.src.classes) );
    U = cell( 1 , length(db.src.classes) );
	for class = 1:length(db.src.classes)
		%% Determine the objects belonging to class.
		ind_obj = find([db.src.objects.class]==class & train_mask);
		
        if isempty(ind_obj)
            %% Class has no objects, skip.
            continue
        end

		%% Calculate centroid and all the principal components.
		mu{class} = mean(db.features( ind_obj , : ));
        X = db.features( ind_obj , : ) - ones( size(db.features( ind_obj , : ),1) , 1 ) * mu{class};
		[U{class} , ~ ] = pca( X );

		%% Truncate principal components if they exceed the maximum dimension.
		if size(U{class},2) > max(opt.dim)
			U{class} = U{class}(:,1:max(opt.dim));
		end
	end
	
	%% Prepare output.
	model.model_type = 'affine';
	model.dim = opt.dim;
	model.mu = mu;
	model.principal_components = U;
end

function mu = mean( X )
	
    N = size(X,1);
    
    mu = ones( 1 , N ) * X/N;
end

function [U,s] = pca( X )
	%% Calculate the principal components of x along the second dimension.
    
    [ U , spectral_matrix ] = eig( X'*X );
    [ s ,ind ] = sort( diag( spectral_matrix ),'descend' );
    U = U(:,ind);
	
end
