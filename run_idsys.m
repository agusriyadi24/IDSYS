
%% run_idsys.m
% Identifikasi sistem (linear & nonlinear) dengan data internal MATLAB: iddata1 (z1)

clear; close all; clc;

%% 0) Muat data internal iddata1 -> z1 (objek iddata)
% Dokumen menunjukkan cara muat: load iddata1 z1; (SISO time-domain)  ([1](https://www.mathworks.com/help/ident/ref/iddata.plot.html))
load iddata1 z1;

% Plot data mentah (top: output, bottom: input)  ([1](https://www.mathworks.com/help/ident/ref/iddata.plot.html))
figure('Name','Data iddata1 (z1)'); plot(z1); grid on;

%% 1) Preprocessing & split train/validation
z1_dt = detrend(z1);                                 % buang offset/trend   ([3](https://www.mathworks.com/help/ident/ug/ways-to-prepare-data-for-system-identification.html))
N = length(z1_dt.y);
Nv_est = round(0.6*N);                               % 60% untuk estimasi
z_est = z1_dt(1:Nv_est);
z_val = z1_dt(Nv_est+1:N);

%% 2) IDENTIFIKASI MODEL LINEAR
models_lin = struct();

% Kandidat ARX/OE (coba beberapa orde kecil-menengah)
models_lin.ARX_221 = arx(z_est, [2 2 1]);           % na nb nk
models_lin.ARX_331 = arx(z_est, [3 3 1]);
models_lin.OE_221  = oe(z_est,  [2 2 1]);           % nb nf nk
models_lin.OE_331  = oe(z_est,  [3 3 1]);
% State-space (orde 2 & 3)
models_lin.SS_2    = ssest(z_est, 2);
models_lin.SS_3    = ssest(z_est, 3);
% Transfer function (2 & 3 pole)
models_lin.TF_2    = tfest(z_est, 2);
models_lin.TF_3    = tfest(z_est, 3);

% Hitung fit (%) pada data validasi (NRMSE)  ([4](https://www.mathworks.com/help/ident/ref/compare.html))
fn_lin = fieldnames(models_lin);
fit_lin = struct();
for i = 1:numel(fn_lin)
    mdl = models_lin.(fn_lin{i});
    [~, fitpct] = compare(z_val, mdl);              % mengembalikan fit (%)
    fit_lin.(fn_lin{i}) = fitpct;
end

% Pilih linear terbaik
[bestFitLin, idxBestLin] = max(cellfun(@(n) fit_lin.(n), fn_lin));
bestNameLin  = fn_lin{idxBestLin};
bestModelLin = models_lin.(bestNameLin);

fprintf('\n=== HASIL LINEAR (iddata1) ===\n');
fprintf('Model linear terbaik: %s, Fit = %.2f%%\n', bestNameLin, bestFitLin);
present(bestModelLin);                               % ringkasan model

figure('Name','Perbandingan model linear terbaik (iddata1)');
compare(z_val, bestModelLin); grid on;

%% 3) IDENTIFIKASI MODEL NONLINEAR
models_nl = struct();

% NARX (sigmoid & wavelet)
models_nl.NARX_sig_221  = nlarx(z_est, [2 2 1], sigmoidnet('NumberOfUnits',8));
models_nl.NARX_sig_331  = nlarx(z_est, [3 3 1], sigmoidnet('NumberOfUnits',10));
models_nl.NARX_wave_221 = nlarx(z_est, [2 2 1], wavenet('NumberOfUnits',10));
% Hammerstein–Wiener: linear [2 2 1] + nonlin statik
models_nl.HW_sat_dead   = nlhw(z_est, [2 2 1], saturation, deadzone);
models_nl.HW_sig_sig    = nlhw(z_est, [2 2 1], sigmoidnet('NumberOfUnits',8), sigmoidnet('NumberOfUnits',8));
