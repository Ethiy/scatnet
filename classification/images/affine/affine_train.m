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
	opt = fill_struct(opt,'dim',80);
	
	%% Create mask to separate the training vectors
	train_mask = ismember(1:length(db.src.objects),train_set);
	
    mu = cell( 1 , length(db.src.classes) );
    v = cell( 1 , length(db.src.classes) );
	for class = 1:length(db.src.classes)
		%% Determine the objects belonging to class.
		ind_obj = find([db.src.objects.class]==class & train_mask);
		
        if isempty(ind_obj)
            %% Class has no objects, skip.
            continue
        end

		%% Calculate centroid and all the principal components.
		mu{class} = sig_mean(db.features(:,ind_obj));
		v{class} = sig_pca(db.features(:,ind_obj),0);

		%% Truncate principal components if they exceed the maximum dimension.
		if size(v{class},2) > max(opt.dim)
			v{class} = v{class}(:,1:max(opt.dim));
		end
	end
	
	%% Prepare output.
	model.model_type = 'affine';
	model.dim = opt.dim;
	model.mu = mu;
	model.v = v;
end

function mu = sig_mean(x)
	%% Calculate mean along second dimension.

    C = size(x,2);
    
    mu = x*ones(C,1)/C;
end

function [u,s] = sig_pca(x,M)
	%% Calculate the principal components of x along the second dimension.

	if nargin > 1 && M > 0
		%% If M is non-zero, calculate the first M principal components.
	    [u,s,v] = svds(x-sig_mean(x)*ones(1,size(x,2)),M);
	    s = abs(diag(s)/sqrt(size(x,2)-1)).^2;
	else
		%% Otherwise, calculate all the principal components.
		[u,d] = eig(cov(x'));
		[s,ind] = sort(diag(d),'descend');
		u = u(:,ind);
	end
end
