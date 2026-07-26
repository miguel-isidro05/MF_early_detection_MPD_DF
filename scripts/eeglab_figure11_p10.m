% Reproduce the Figure 11 processing path with EEGLAB itself.
% Usage from the project root:
% /Applications/MATLAB_R2025b.app/bin/matlab -batch "run('scripts/eeglab_figure11_p10.m')"
%
% This is a diagnostic reproduction. The paper specifies EEGLAB ICA but does
% not publish ICA weights, rejected components, reference, or PSD options.

clear; clc;

project_root = fileparts(fileparts(mfilename('fullpath')));
eeglab_root = '/Users/miguelisidrobaez/Desktop/Apps/eeglab2026.0.0';
raw_root = fullfile(project_root, '..', 'Dataset', '28455737', 'Raw Dataset');
eeg_path = fullfile(raw_root, 'EEG', 'MPDDF_raw_10_EEG.edf');
annotation_path = fullfile(raw_root, 'Annotation', 'MPDDF_raw_10_Annotation.txt');
output_dir = fullfile(project_root, 'reproduction', 'eeglab_figure11_p10');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% Verified against the official alignment audit for Participant 10.
aligned_start_clock_sec = 51146;
eeg_offset_sec = 11;
aligned_duration_sec = 8101;
edge_trim_sec = 60;
artifact_probability = 0.80;
rng(42, 'twister');

addpath(eeglab_root);
[ALLEEG, EEG, CURRENTSET] = eeglab('nogui'); %#ok<ASGLU>
EEG = pop_biosig(eeg_path, 'importevent', 'off', 'importannot', 'off');
EEG = eeg_checkset(EEG);
EEG = pop_chanedit(EEG, 'lookup', fullfile(eeglab_root, 'functions', 'supportfiles', 'Standard-10-5-Cap385.sfp'));

% Figure 6 preprocessing, explicitly reported by the data descriptor.
EEG_visual = pop_eegfiltnew(EEG, 0.3, 35, [], 0, 0, 0, 0, 1);
EEG_visual = pop_eegfiltnew(EEG_visual, 49, 51, [], 1, 0, 0, 0, 1);

% EEGLAB best practice: fit ICA on a 1 Hz high-passed copy, then transfer
% weights to the Figure 6-preprocessed data before subtracting components.
EEG_ica = pop_eegfiltnew(EEG, 1, [], [], 0, 0, 0, 0, 1);
EEG_ica = pop_eegfiltnew(EEG_ica, 49, 51, [], 1, 0, 0, 0, 1);
EEG_ica = pop_runica(EEG_ica, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off');
EEG_ica = pop_iclabel(EEG_ica, 'default');

classes = EEG_ica.etc.ic_classification.ICLabel.classes;
probabilities = EEG_ica.etc.ic_classification.ICLabel.classifications;
artifact_classes = ismember(classes, {'Muscle', 'Eye'});
rejected = find(any(probabilities(:, artifact_classes) >= artifact_probability, 2))';

EEG_visual.icaweights = EEG_ica.icaweights;
EEG_visual.icasphere = EEG_ica.icasphere;
EEG_visual.icawinv = EEG_ica.icawinv;
EEG_visual.icachansind = EEG_ica.icachansind;
EEG_clean = pop_subcomp(EEG_visual, rejected, 0);

% Build the official aligned, 1-minute-edge-trimmed 30-second homogeneous blocks.
labels = read_labels(annotation_path, aligned_start_clock_sec, aligned_duration_sec);
segment_starts = cell(1, 5);
for start_sec = edge_trim_sec:30:(aligned_duration_sec - edge_trim_sec - 30)
    block = labels(start_sec + 1:start_sec + 30);
    if numel(unique(block)) == 1 && block(1) >= 0 && block(1) <= 4
        segment_starts{block(1) + 1}(end + 1) = start_sec; %#ok<SAGROW>
    end
end

results = struct();
for label = 0:4
    starts = segment_starts{label + 1};
    frames = round(30 * EEG_clean.srate);
    data = zeros(EEG_clean.nbchan, frames, numel(starts), 'single');
    for index = 1:numel(starts)
        first = round((eeg_offset_sec + starts(index)) * EEG_clean.srate) + 1;
        data(:, :, index) = EEG_clean.data(:, first:first + frames - 1);
    end
    [spectra_db, freqs] = spectopo(data, frames, EEG_clean.srate, ...
        'plot', 'off', 'winsize', round(EEG_clean.srate), 'overlap', 0, 'rmdc', 'on');
    frequency_mask = freqs >= 0.3 & freqs <= 35;
    results(label + 1).label = label;
    results(label + 1).n_segments = numel(starts);
    results(label + 1).mean_db = mean(spectra_db(:, frequency_mask), 2);
    results(label + 1).sum_linear_uv2_hz = sum(10 .^ (spectra_db(:, frequency_mask) / 10), 2);
end

write_outputs(results, EEG_clean.chanlocs, rejected, classes, probabilities, output_dir, artifact_probability);

function labels = read_labels(path, aligned_start_clock_sec, duration_sec)
    lines = readlines(path);
    times = zeros(numel(lines), 1);
    values = zeros(numel(lines), 1);
    for index = 1:numel(lines)
        parts = split(strtrim(lines(index)), ',');
        clock = sscanf(parts(1), '%d:%d:%d');
        times(index) = clock(1) * 3600 + clock(2) * 60 + clock(3);
        values(index) = str2double(parts(end));
    end
    labels = zeros(duration_sec, 1);
    for index = 1:numel(times)
        first = max(1, times(index) - aligned_start_clock_sec + 1);
        if index < numel(times)
            last = min(duration_sec, times(index + 1) - aligned_start_clock_sec);
        else
            last = duration_sec;
        end
        if first <= last, labels(first:last) = values(index); end
    end
end

function write_outputs(results, chanlocs, rejected, classes, probabilities, output_dir, threshold)
    names = {'Wakefulness', 'Fatigue1', 'Fatigue2', 'Fatigue3', 'Fatigue4'};
    save(fullfile(output_dir, 'figure11_eeglab_record.mat'), ...
        'results', 'rejected', 'classes', 'probabilities', 'threshold');
    writematrix(probabilities, fullfile(output_dir, 'iclabel_probabilities.csv'));
    writecell(classes, fullfile(output_dir, 'iclabel_classes.csv'));
    writematrix(rejected, fullfile(output_dir, 'rejected_components.csv'));

    fields = {'mean_db', 'sum_linear_uv2_hz'};
    filenames = {'figure11_eeglab_mean_db.png', 'figure11_eeglab_sum_linear.png'};
    for variant = 1:numel(fields)
        values = cat(2, results.(fields{variant}));
        figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1400 850]);
        limits = [min(values, [], 'all'), max(values, [], 'all')];
        for label = 1:5
            subplot(2, 3, label);
            topoplot(values(:, label), chanlocs, 'maplimits', limits, 'electrodes', 'on');
            title(sprintf('%c  %s (n=%d)', char(96 + label), names{label}, results(label).n_segments));
        end
        subplot(2, 3, 6); axis off;
        colormap(parula); colorbar('Position', [0.92 0.15 0.02 0.7]);
        sgtitle(sprintf('Participant 10 | EEGLAB runica + ICLabel >= %.2f', threshold));
        exportgraphics(gcf, fullfile(output_dir, filenames{variant}), 'Resolution', 300);
        close(gcf);
    end
end
