% WAVELET_2D Compute the wavelet transform of a signal x
%
% Usage
%    [x_phi, x_psi] = WAVELET_2D(x, filters, options)
%
% Input
%    x (numeric): the input signal
%    filters (cell): cell containing the filters
%    options (structure): options of the wavelet transform
%
% Output
%    x_phi (numeric): Low pass part of the wavelet transform
%    x_psi (cell): Wavelet coeffcients of the wavelet transform
%    meta_phi (struct): meta associated to x_phi
%    meta_psi (struct): meta assocaited to y_phi
%
% Description
%    WAVELET_2D computes a wavelet transform, using the signal and the
%    filters in the Fourier domain. The signal is padded in order to avoid
%    border effects.
%
%    The meta information concerning the signal x_phi, x_psi(scale, angle,
%    resolution) can be found in meta_phi and meta_psi.
%
% See also
%   WAVELET_2D_PYRAMID, CONV_SUB_2D, WAVELET_FACTORY_2D_PYRAMID

function [x_phi, x_psi, meta_phi, meta_psi] = wavelet_2d(x, filters, options)
    %% Options
    if(nargin<3)
        options = struct;
    end
    white_list = {'resolution', 'psi_mask', 'oversampling', 'precision' };
    check_options_white_list(options, white_list);
    options = fill_struct(options, 'resolution', 0);
    options = fill_struct(options, 'oversampling', 1);
    options = fill_struct(options , 'precision' , 'double' );
    options = fill_struct(options, 'psi_mask', ones(1,numel(filters.psi.filter)));
    
    oversampling = options.oversampling;
    psi_mask = options.psi_mask;
    precision = options.precision;
    
    %% Padding and Fourier transform
    size_paded = filters.meta.size_filter' / 2^options.resolution;
    if strcmp( precision ,  'single')
        x = single(x);
    end
    x_fourier = fft2(pad_signal(x, size_paded, []));
    
    %% Low-pass filtering, downsampling and unpadding
    Q = filters.meta.Q;
    J = filters.phi.meta.J;
    downsampling_rate = max(floor(J/Q)- options.resolution - oversampling, 0); % downsampling rate
    x_phi = real(  conv_sub_2d( x_fourier , filters.phi.filter , downsampling_rate )  );
    x_phi = unpad_signal( x_phi , downsampling_rate*[1 1] , size(x));
    
    meta_phi.j = -1;
    meta_phi.theta = -1;
    meta_phi.resolution = options.resolution + downsampling_rate;
    
    %% Band-pass filtering, downsampling and unpadding
    indices = find(psi_mask);
    x_psi = cell( 1 , length(indices) ) ;
    meta_psi = struct();
    for p = indices
        j = filters.psi.meta.j(p);
        downsampling_rate = max(floor(j/Q)- options.resolution - oversampling, 0);
        x_psi{p} = conv_sub_2d(x_fourier, filters.psi.filter{p}, downsampling_rate);
        x_psi{p} = unpad_signal(x_psi{p}, downsampling_rate*[1 1], size(x));
        meta_psi.j(1,p) = filters.psi.meta.j(p);
        meta_psi.theta(1,p) = filters.psi.meta.theta(p);
        meta_psi.resolution(1,p) = options.resolution + downsampling_rate;
    end
    
end
