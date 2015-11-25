% FORMAT_SCAT Formats a scattering representation
%
% Usages
%    [out,meta]  = FORMAT_SCAT(S)
%
%    [out, meta] = FORMAT_SCAT(S, fmt)
%
% Input
%    S (cell): The scattering representation to be formatted.
%    fmt (string): The desired format. Can be either 'raw',
%     'order_table', 'vector' or 'table' (default 'table').
%
% Output
%    out: The scattering representation in the desired format (see below).
%    meta (struct): Properties of the scattering nodes in out.
%
% Description
%    Three different formats are available for the scattering transform:
%       'raw': Does nothing, just return S. The meta structure is empty.
%       'order_table': For each order, creates a table of scattering
%          coefficients with scattering index running along the first dimen-
%          sion, time/space along the second, and signal index along the
%          third. The out variable is then a cell array of tables, while
%          the meta variable is a cell array of meta structures, each
%          corresponding to the meta structure for the given order.
%       'table': Same as 'order_table', but with the tables for each order
%          concatenated into one table, which is returned as out. Note that
%          this requires that each order is of the same resolution, that is
%          that the lowpass filter phi of each filter bank is of the same
%          bandwidth. The meta variable is one meta structure formed by con-
%          catenating the meta structure of each order and filling out with
%          -1 where necessary (the j field, for example).
%       'vector': transforms the 'table' output into a matrix for images
%
% See also
%   FLATTEN_SCAT, REORDER_SCAT

function [out,meta] = format_scat(X,fmt)
    if nargin < 2
        fmt = 'table';
    end
    
    switch fmt
        case 'raw'
        %% Case : raw
            out = X;
            meta = [];
            return
            
        case 'table'
        %% Case : table
            non_empties = cellfun(@(x) ~isempty(x.signal),X);
            resolution = cellfun(@(x) length(x.signal{1}),X(non_empties));
            % if not all nonzero resolutions are equal, an error is thrown
            if ~all(nonzeros(resolution)==resolution(1))
                error(['To use ''table'' output format, all orders ' ...
                    'must be of the same resolution. Consider' ...
                    'using the ''order_table'' output format.']);
            end
            X = flatten_scat(X); % puts all layers together
            if ~isempty(X{1}.signal)
                out = zeros( ...
                [length(X{1}.signal) size(X{1}.signal{1})], ...
                'like' , X{1}.signal{1} );

                for j = 0:length(X{1}.signal)-1
                    out(1+j,1:numel(X{1}.signal{1})) = ...
                    X{1}.signal{1+j}(:);
                end
            end
            meta = X{1}.meta;
            return
                
        case 'order_table'
        %% Case : order table
            M = length(X); % M equals 1 if X has been flattened
            out = cell(1,M);
            meta = cell(1,M);

            for m = 0:M-1
                if isempty(X{1+m}.signal)
                    out{1+m} = [];
                else
                    out{1+m} = zeros( ...
                    [length(X{1+m}.signal) size(X{1+m}.signal{1})], ...
                    'like' , X{1+m}.signal{1} );

                    for j = 0:length(X{1+m}.signal)-1
                        out{1+m}(1+j,1:numel(X{1+m}.signal{1})) = ...
                        X{1+m}.signal{1+j}(:);
                    end
                end
                meta{m+1} = X{m+1}.meta;
            end
            return
        case 'vector'
        %% Case : vector           
            non_empties = cellfun(@(x) ~isempty(x.signal),X);
            resolution = cellfun(@(x) length(x.signal{1}),X(non_empties));
            % if not all nonzero resolutions are equal, an error is thrown
            if ~all(nonzeros(resolution)==resolution(1))
                error(['To use ''table'' output format, all orders ' ...
                    'must be of the same resolution. Consider' ...
                    'using the ''order_table'' output format.']);
            end
            
            X = flatten_scat(X); % puts all layers together
            if ~isempty(X{1}.signal)
                out = zeros( 1 , length(X{1}.signal) * numel(X{1}.signal{1}), ...
                'like' , X{1}.signal{1} );
                
                r = 1;
                for j = 0:length(X{1}.signal)-1
                    index = r:r +  numel( X{1}.signal{1+j} ) - 1;
                    out( index ) = X{1}.signal{1+j}(:);
                    r = r + numel( X{1}.signal{1+j} );
                end
            end
            meta = X{1}.meta;
            return
            
        otherwise
        %% Case : non identified
            error(['Unknown format. Available formats are ''raw'', ''table'''...
            ' or ''order_table''.']);
    end
    
end
