% CALC_TRAIN_WEIGHTS calculates training weights for the SVM
%
% Usage
%    db_weights = calc_train_weights(db,train_set)
%
% Input
%    db (struct): The database containing the feature vector.
%    train_set (int): The object indices of the training instances.
%    
%
% Output
%    db_weights (string): The training weights.
%
% Description
%     The weight of each class is the ratio between the total number of
%     training features Nfeat_tot, and the number of training features of 
%     the class Nfeat_train_k ie w_k =  Nfeat_tot/Nfeat_train_k
%     Note that the range of the values of C used for cross_validation 
%     should take these weights into consideration.
%
% See also
%    SVM_TRAIN

function db_weights = calc_train_weights(db,train_set)
        
    db_weights = cell( 1 , length(db.src.classes) ) ;
    
    number_train_features = length( train_set );

    for class = 1:length(db.src.classes)
            index = find([db.src.objects.class] == class);
            mask_class = ismember( index, train_set );
            index = index( mask_class > 0);
            db_weights{ class } = [ ' -w' num2str( class ) ' ' num2str( number_train_features/numel(index) )];
    end

    db_weights = [db_weights{:}];

end
