% WAVELET_LAYER_1D Compute the one-dimensional wavelet transform from
% the modulus wavelet coefficients of the previous layer.
%
% Usages
%    [U_phi , U_psi] = wavelet_layer_1d(U, filters)
%
%    [U_phi , U_psi] = wavelet_layer_1d(U, filters, scat_opt)
%
%    [U_phi , U_psi] = wavelet_layer_1d(U, filters, scat_opt, wavelet)
%
% Input
%    U (struct): The input layer to be transformed.
%    filters (cell): The filters of the wavelet transform.
%    scat_opt (struct): The options of the wavelet layer. Some are used in the
%       function itself, while others are passed on to the wavelet transform.
%       The parameters used by WAVELET_LAYER_1D are:
%          path_margin: The margin used to determine wavelet decomposition
%             scales with respect to the bandwidth of the signal. If the band-
%             with of a signal in U is bw, only wavelet filters of center fre-
%             quency less than bw*2^path_margin are applied (default 0).
%    wavelet (function handle): the wavelet transform function (default
%       @wavelet_1d).  
%
% Output
%    U_phi The coefficients of in, lowpass-filtered (scattering
%       coefficients).
%    U_psi: The wavelet transform coefficients.
%
% Description
%    This function has a pivotal role between WAVELET_1D (which computes a
%    single wavelet transform), and WAVELET_FACTORY_1D (which creates the
%    whole cascade). Given inputs modulus wavelet coefficients
%    corresponding to a layer, WAVELET_LAYER_1D computes the wavelet
%    transform coefficients of the next layer using WAVELET_1D. 
% See also
%   WAVELET_1D, WAVELET_FACTORY_1D, WAVELET_LAYER_2D

function [U_phi, U_psi] = wavelet_layer_1d(U, filters, scat_opt, wavelet)
	%% Options
    if nargin < 3
        scat_opt = struct();
    end
	
	if nargin < 4
		wavelet = @wavelet_1d;
	end
	
	scat_opt = fill_struct(scat_opt, 'path_margin', 0);

	[ psi_xi , ~ , ~ ] = filter_freq( filters.meta );
	
    if ~isfield(U.meta, 'bandwidth')
        U.meta.bandwidth = 2*pi;
    end
    if ~isfield(U.meta, 'resolution')
        U.meta.resolution = 0;
    end
	
	U_phi.signal = {};
	U_phi.meta.bandwidth = [];
	U_phi.meta.resolution = [];
	U_phi.meta.j = zeros(size(U.meta.j,1),0);
	
	U_psi.signal = {};
	U_psi.meta.bandwidth = [];
	U_psi.meta.resolution = [];
	U_psi.meta.j = zeros(size(U.meta.j,1)+1,0);
    
    %% Number of outputs
    % do not compute any convolution
    % with psi if the user does get U_psi
	calc_U = (nargout>=2);
	
    %% Scattering layer : convolving with psi and phi
	r = 1;
	for p = 1:length( U.signal ) % for every signal
		current_bandwidth = U.meta.bandwidth( p ) * 2^scat_opt.path_margin;
		psi_mask = calc_U & (current_bandwidth > psi_xi); % if not true it will not convolve
		
		scat_opt.x_resolution = U.meta.resolution( p );
		scat_opt.psi_mask = psi_mask;
		[x_phi, x_psi, meta_phi, meta_psi] = wavelet( U.signal{p} , filters, scat_opt);
		
		U_phi.signal{1,p} = x_phi;
		U_phi.meta = map_meta(U.meta,p,U_phi.meta,p);
		U_phi.meta.bandwidth(1,p) = meta_phi.bandwidth;
		U_phi.meta.resolution(1,p) = meta_phi.resolution;
		
		ind = r:r+sum(psi_mask)-1;
		U_psi.signal(1,ind) = x_psi(1,psi_mask);
		U_psi.meta = map_meta(U.meta,p,U_psi.meta,ind,{'j'});
		U_psi.meta.bandwidth(1,ind) = meta_psi.bandwidth(1,psi_mask);
		U_psi.meta.resolution(1,ind) = meta_psi.resolution(1,psi_mask);
		U_psi.meta.j(:,ind) = [U.meta.j(:,p)*ones(1,length(ind)); ...
			meta_psi.j(1,psi_mask)];
			
		r = r+length(ind);
	end
end
