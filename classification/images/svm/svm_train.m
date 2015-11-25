% SVM_TRAIN Train an SVM classifier
%
% Usage
%    model = SVM_TRAIN(db, train_set, options)
%
% Input
%    db (struct): The database containing the feature vector.
%    train_set (int): The object indices of the training instances.
%    options (struct): The training options:
%          options.kernel_type (char): The kernel type: 'linear' or 'gaussian'
%             (default 'gaussian').
%          options.C (numeric): The slack factor (default 8).
%          options.gamma (numeric): The gamma of the Gaussian kernel (default 
%             1e-4).
%          options.no_inplace (boolean): Do not use the inplace version of 
%             LIBSVM, even if it available (default false).
%          options.full_test_kernel (boolean): Explicity calculate the test
%             kernel instead of relying on the precalculated kernel. Used if 
%             the kernel is only defined on the training vectors (default 
%             false).
%          options.reweight (boolean): Add weights to rebalance the training set if 
%             it is imbalanced. The rebalancing is done so that the distribu-
%             tion of the training samples seem to be uniform for all the 
%             classes (default 0).
%
% Output
%    model (struct): The SVM model.
%
% Description
%    The svm_train function provides an interface to the LIBSVM set of
%    SVM training routines. If available, will use the inplace version found
%    in libsvm-compact (see http://www.di.ens.fr/data/software/) to save
%    memory and speed up calculations.
%
% See also
%    SVM_TEST, CLASSIF_ERR, CLASSIF_RECOG, CLAC_TRAIN_WEIGHTS


function model = svm_train( db , train_set , options )

	if nargin < 3
		options = struct();
	end

	% Set default options.
	options = fill_struct(options, 'no_inplace', 0);
	options = fill_struct(options, 'full_test_kernel', 0);

	options = fill_struct(options, 'kernel_type', 'gaussian');

	options = fill_struct(options, 'gamma', 1e-4);
	options = fill_struct(options, 'C', 8);
	options = fill_struct(options, 'reweight', 0);
	options = fill_struct(options, 'b', 0);

	%% Extract feature vector indices of the objects in the training set and their respective classes.
	feature_class = zeros( length( train_set ) , 1 );
	for index = 1:length( train_set )
		feature_class( index ) = db.src.objects( train_set(index) ).class ;
	end

	%% Is there are pre-calculated kernel of the same type as specified in the options?
	precalc_kernel = isfield( db , 'kernel' ) && ...
		strcmp( options.kernel_type , db.kernel.kernel_type );

	%% Slackness parameter is always specified.
	parameters = ['-q -c ' num2str(options.C)];
	if ~precalc_kernel
		%% Non-precalculated kernel - specify type to LIBSVM.
        switch options.kernel_type
            case 'linear'
                parameters = [parameters ' -t 0'];
            case 'gaussian'
            % Gaussian kernel - also give gamma parameter.
                parameters = [parameters ' -t 2 -g ' num2str(options.gamma)];
            otherwise
                error('Unsupported kernel type!');
        end
		%% Feature matrix for LIBSVM is just the submatrix containing the training feature vectors.
		features = db.features( train_set , : );
	else
		%% Precalculated kernel. If inplace version of LIBSVM is available, we pass
		% it the kernel plus a mask, otherwise we extract the relevant parts of the
		% kernel.

		%% If only parts of the training feature vectors are included among the vec-
		% tors in the kernel, use only those.
		[ kernel_mask , kernel_ind ] = ismember( 1:size( db.features , 1 ) , db.kernel.kernel_set );
		train_set = kernel_ind( train_set(kernel_mask( train_set )));

		if exist( 'svmtrain_inplace' , 'file' ) && ~options.no_inplace
			% The inplace version of LIBSVM exists and we can use it.

			%% Calculate the classes for the vectors in the kernel. 
			feature_class = zeros( 1 , size(db.features,1) );
			for index = 1:length( db.src.objects )
				feature_class( index ) = db.src.objects(index).class;
			end
			feature_class = feature_class( db.kernel.kernel_set );

			%% Send the whole kernel to LIBSVM.
			features = db.kernel.K;

			if strcmp(db.kernel.kernel_type,'linear') && ...
				strcmp(db.kernel.kernel_format,'square')
				% Feature matrix contains kernel values in square form.
				parameters = [parameters ' -t 4'];
			elseif strcmp(db.kernel.kernel_type,'linear') && ...
				strcmp(db.kernel.kernel_format,'triangle')
				% Feature matrix contains kernel values in triangular form.
				parameters = [parameters ' -t 5'];
			elseif strcmp(db.kernel.kernel_type,'gaussian') && ...
				strcmp(db.kernel.kernel_format,'square')
				% Feature matrix contains \|x_i-x_j\|^2 in square form. To obtain a Gauss-
				% ian kernel, LIBSM thus needs to multiply by -gamma and exponentiate.
				parameters = [parameters ' -t 6 -g ' num2str(options.gamma)];
			elseif strcmp(db.kernel.kernel_type,'gaussian') && ...
				strcmp(db.kernel.kernel_format,'triangle')
				% Same as above, but in triangular form.
				parameters = [parameters ' -t 7 -g ' num2str(options.gamma)];
			else
				error('Unknown kernel type/format!');
			end
		else
			%% We don't have the inplace version of LIBSVM. 
			parameters = [parameters ' -t 4'];

			if strcmp(db.kernel.kernel_format, 'triangle')
				error(['Triangular kernels not supported for standard ' ...
					' LIBSVM version. Please try libsvm-compact.'])
			end
			
			% Send the part of the kernel containing the training vector columns.
			features = db.kernel.K( train_set , : );

			if strcmp( db.kernel.kernel_type , 'gaussian' )
				% Since this version of LIBSVM doesn't support on-the-fly exponentiation,
				% we calculate the correct Gaussian kernel here.
				features(2:end,:) = exp(-options.gamma*features(2:end,:));
				parameters = [parameters ' -g ' num2str(options.gamma)];
			end
		end
	end

	if options.reweight
		%% If reweighting to obtain uniform distribution is needed, add the weights.
		db_weights = calc_train_weights(db, train_set);
		parameters = [parameters db_weights];
	end

	%% Probability outputs?
	parameters = [parameters ' -b ' num2str(options.b)];

	%% Are we to calculate complete kernel when testing?
	model.full_test_kernel = options.full_test_kernel;

	%% Which vectors were used to train the SVM?
	model.train_set = train_set;

	%% Call the desired LIBSVM routine.
	if options.no_inplace || ~exist('svmtrain_inplace' , 'file' )
		model.svm = svmtrain(double(feature_class), ...
			double(features),parameters);
	else
		% To specify the training vectors, ind_features is passed as a mask.
		model.svm = svmtrain_inplace(feature_class, ...
			single(features),parameters,train_set);
	end
end