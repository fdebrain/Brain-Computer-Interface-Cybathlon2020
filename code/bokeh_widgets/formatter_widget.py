import logging
from collections import Counter
import traceback

import numpy as np
import mne
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import (Div, Select, Button, Slider, CheckboxButtonGroup,
                          Span, Label, Toggle, ColumnDataSource)
from bokeh.layouts import column, row

from config import main_config
from src.vhdr_formatter import format_session, load_h5, load_vhdr


class FormatterWidget:
    def __init__(self):
        self.data_path = main_config['data_path']
        self.labels = list(main_config['pred_decoding'].values())
        self.channel2idx = {}
        self._t = 0
        self._data = dict()
        self.source = ColumnDataSource(data=dict(timestamps=[], values=[]))
        self.source_events = ColumnDataSource(data=dict(events=[], actions=[]))
        self.play_rate_ms = 100

    @property
    def speed(self):
        return self.slider_speed.value * self.play_rate_ms / 1000

    @property
    def fs(self):
        return self._data.get('fs', -1)

    @property
    def t_max(self):
        return len(self._data.get('timestamps', [0]))

    @property
    def win_len(self):
        return self.slider_win_len.value

    @property
    def t(self):
        ''' Current visible time in seconds'''
        return self._t

    @t.setter
    def t(self, val):
        assert val < self.t_max, 'End of signal'

        self._t = val
        self.slider_t.value = val

        # Extract ROI signal (convert in timestamp units)
        start = int(self.t * self.fs)
        end = start + int(self.win_len * self.fs)
        ts = self._data['timestamps'][start:end]
        eeg = self._data['values'][self.channel_idx, start:end]

        # Extract ROI events
        roi_events = np.array([(t, action)
                               for t, action in self._data['events']
                               if start < t*self.fs < end]).reshape((-1, 2))

        if len(roi_events > 0):
            self.event_marker.visible = True
            self.event_marker.location = roi_events[0, 0]
            self.event_label.visible = True
            self.event_label.x = roi_events[0, 0]
            self.event_label.text = str(int(roi_events[0, 1]))

        # Update source
        self.source.data = dict(timestamps=ts, values=eeg)
        self.source_events = dict(events=roi_events[:, 0],
                                  actions=roi_events[:, 1])

    @property
    def available_pilots(self):
        pilots = self.data_path.glob('*')
        return [''] + [p.parts[-1] for p in pilots]

    @property
    def selected_pilot(self):
        return self.select_pilot.value

    @property
    def available_sessions(self):
        pilot_path = self.data_path / self.selected_pilot
        sessions = pilot_path.glob('*')
        return [''] + [s.parts[-1] for s in sessions]

    @property
    def selected_session(self):
        return self.select_session.value

    @property
    def session_path(self):
        return self.data_path / self.selected_pilot / self.selected_session

    @property
    def run_type(self):
        return 'game' if self.is_game_session else 'vhdr'

    @property
    def session_runs(self):
        regex = '*.h5' if 'HDF5' in self.selected_settings else '*.vhdr'
        return list(self.session_path.glob(f'{self.run_type}/{regex}'))

    @property
    def available_runs(self):
        return [''] + [r.name for r in self.session_runs]

    @property
    def run_path(self):
        return self.session_path / self.run_type / self.select_run.value

    @property
    def channel_name(self):
        return self.select_channel.value

    @property
    def channel_idx(self):
        return self.channel2idx.get(self.channel_name, None)

    @property
    def labels_encoding(self):
        markers = [int(s.value) for s in self.select_labels]
        return dict(zip(markers, [0, 1, 2, 3]))

    @property
    def labels_decoding(self):
        markers = [int(s.value) for s in self.select_labels]
        return dict(zip(self.labels, markers))

    @property
    def pre(self):
        return self.slider_pre_event.value

    @property
    def post(self):
        return self.slider_post_event.value

    @property
    def selected_settings(self):
        active = self.checkbox_settings.active
        return [self.checkbox_settings.labels[i] for i in active]

    @property
    def should_balance(self):
        return 'Balance' in self.selected_settings

    @property
    def should_preprocess(self):
        return 'Preprocess' in self.selected_settings

    @property
    def is_game_session(self):
        return 'Game session' in self.selected_settings

    def update_widget(self):
        self.select_pilot.options = self.available_pilots
        self.select_session.options = self.available_sessions
        self.select_run.options = self.available_runs
        self.button_format.button_type = "primary"
        self.button_format.label = "Format"

    def on_extract_change(self, attr, old, new):
        logging.info(
            f'Epochs extracted ({self.pre},{self.post}) around marker')
        self.button_format.button_type = "primary"
        self.button_format.label = "Format"

    def on_pilot_change(self, attr, old, new):
        logging.info(f'Select pilot {new}')
        self.update_widget()
        self.select_session.value = ''

    def on_session_change(self, attr, old, new):
        logging.info(f'Select session {new}')
        self.update_widget()

        # Get session info
        fs = 0
        duration = 0
        events_counter = dict()
        n_channels, n_samples = 0, 0
        for run in self.session_runs:
            if run.suffix == '.h5':
                continue

            raw = mne.io.read_raw_brainvision(vhdr_fname=run,
                                              preload=False,
                                              verbose=False)

            fs = int(raw.info['sfreq'])
            n_channels = len(raw.ch_names)
            n_samples += raw.n_times
            duration += int(raw.n_times/(fs*60))
            events = mne.events_from_annotations(raw, verbose=False)[0]

            # Extract only decimal digit if game session
            if self.is_game_session:
                events = Counter([int(str(e[-1])[0]) for e in events])
            else:
                events = Counter([e[-1]for e in events])

            for event in events:
                events_counter[event] = events_counter.get(event, 0) + \
                    events[event]

        # Update displayed info
        self.div_info.text = f'<b>Sampling frequency</b>: {fs} Hz <br>'
        self.div_info.text += f'<b>Nb of channels</b>: {n_channels} <br>'
        self.div_info.text += f'<b>Nb of samples</b>: {n_samples} '
        self.div_info.text += f'({duration} mn) <br>'
        self.div_info.text += f'<b>Nb of occurence per events:</b><br>'
        for event, count in events_counter.items():
            if count > 5:
                self.div_info.text += f'&emsp; <b>{event}:</b>  {count} <br>'

    def on_run_change(self, attr, old, new):
        logging.info(f'Select visualizer run {new}')
        self.select_channel.value = ''
        self.update_widget()

        # Load eeg file
        if self.run_path.suffix == '.vhdr':
            raw = load_vhdr(self.run_path)
        else:
            raw = load_h5(self.run_path)

        # Get channels
        available_channels = raw.ch_names
        self.select_channel.options = [''] + available_channels
        self.channel2idx = {c: i+1 for i, c in enumerate(available_channels)}

        # Get events
        events = mne.events_from_annotations(raw, verbose=False)[0]

        # Store signal and events
        self._data['fs'] = raw.info['sfreq']
        self._data['values'], self._data['timestamps'] = raw.get_data(
            return_times=True)
        self._data['events'] = [(ts/self.fs, action)
                                for ts, action in events[:, [0, 2]]]

        self._t = 0

    def on_settings_change(self, attr, old, new):
        logging.info(f'Changed settings: {new}')
        if 2 in new or 3 in new:
            self.update_widget()
            self.on_session_change('attr',
                                   self.selected_session,
                                   self.selected_session)

    def on_format_start(self):
        assert self.select_session.value != '', \
            'Select a session to format first !'
        self.button_format.button_type = "warning"
        self.button_format.label = "Formatting..."
        curdoc().add_next_tick_callback(self.on_format)

    def on_format(self):
        remove_ch = ['Fp1', 'Fp2']
        extraction_settings = dict(pre=self.pre, post=self.post,
                                   marker_decodings=self.labels_decoding)
        preprocess_settings = dict(resample=False,
                                   preprocess=self.should_preprocess,
                                   remove_ch=remove_ch)

        try:
            format_session(self.session_runs,
                           self.session_path,
                           extraction_settings,
                           preprocess_settings,
                           self.labels_encoding,
                           self.is_game_session,
                           self.should_balance)
        except Exception:
            logging.info(f'Failed to format - {traceback.format_exc()}')
            self.button_format.button_type = "danger"
            self.button_format.label = "Failed"
            return

        self.button_format.button_type = "success"
        self.button_format.label = "Formatted"

    def on_channel_change(self, attr, old, new):
        logging.info(f'Select channel {new}')

        if self.channel_idx is not None:
            # Update source
            start = int(self.t * self.fs)
            end = start + int(self.win_len * self.fs)
            ts = self._data['timestamps'][start:end]
            eeg = self._data['values'][self.channel_idx, start:end]
            self.source.data = dict(timestamps=ts, values=eeg)

            # Update plot axis label
            self.plot_signal.yaxis.axis_label = f'Amplitude - {self.channel_name}'

            # Update navigation slider
            self.slider_t.end = int(self.t_max / self.fs)

    def on_win_len_change(self, attr, old, new):
        # logging.info(f'Win_len update: {new}')
        self.t = self._t

    def on_t_change(self, attr, old, new):
        # logging.info(f't update: {new}')
        self.t = new

    def callback_play(self):
        shift = int(self.speed * self.fs)
        start = int((self.t + self.win_len) * self.fs)
        end = start + shift

        if self.speed > 0:
            ts = np.roll(self.source.data['timestamps'], -shift)
            ts[-shift:] = self._data['timestamps'][start:end]
            eeg = np.roll(self.source.data['values'], -shift, axis=-1)
            eeg[-shift:] = self._data['values'][self.channel_idx, start:end]
            self.source.data = dict(timestamps=ts, values=eeg)
            self.t += self.speed

    def on_toggle_play(self, active):
        assert self.channel_name != '', 'Select a channel first!'
        if active:
            logging.info('Play')
            self.play_toggle.button_type = 'warning'
            self.callback = curdoc().add_periodic_callback(self.callback_play,
                                                           self.play_rate_ms)
        else:
            logging.info('Stop')
            self.play_toggle.button_type = 'primary'
            curdoc().remove_periodic_callback(self.callback)

    def reset_plot(self):
        self.source.data = dict(timestamps=[], values=[])
        self.source_events.data = dict(events=[], actions=[])

    def create_widget(self):
        # Select - Pilot
        self.select_pilot = Select(title='Pilot:',
                                   options=self.available_pilots, )
        self.select_pilot.on_change('value', self.on_pilot_change)

        # Select - Session to format/visualize
        self.select_session = Select(title='Session:')
        self.select_session.on_change('value', self.on_session_change)

        # Select - Label encoding mappings
        self.select_labels = [Select(title=self.labels[id],
                                     options=[str(i) for i in range(30)],
                                     value=str(id+3),
                                     width=80)
                              for id in range(4)]

        # Slider - Extraction window
        self.slider_pre_event = Slider(start=-10, end=10, value=2,
                                       title='Window start (s before event)')
        self.slider_pre_event.on_change('value', self.on_extract_change)
        self.slider_post_event = Slider(start=-15, end=15, value=4,
                                        title='Window end (s after event)')
        self.slider_post_event.on_change('value', self.on_extract_change)

        # Checkbox - Preprocessing
        self.checkbox_settings = CheckboxButtonGroup(
            labels=['Balance', 'Preprocess', 'Game session', 'HDF5'])
        self.checkbox_settings.on_change('active', self.on_settings_change)

        self.button_format = Button(label="Format", button_type="primary")
        self.button_format.on_click(self.on_format_start)

        # Div - Additional informations
        self.div_info = Div()

        # Select - Run to visualize
        self.select_run = Select(title="Run")
        self.select_run.on_change('value', self.on_run_change)

        # Select - Channel to visualize
        self.select_channel = Select(title="Channel")
        self.select_channel.on_change('value', self.on_channel_change)

        # Slider - Navigate through signal
        self.slider_t = Slider(title="Navigate", start=0,
                               end=1000, value=0)
        self.slider_t.on_change('value', self.on_t_change)

        # Slider - Window length
        self.slider_win_len = Slider(title="Win length [s]", start=1,
                                     end=10, value=2)
        self.slider_win_len.on_change('value', self.on_win_len_change)
        # Slider - Play speed rate
        self.slider_speed = Slider(title="Speed", start=0,
                                   end=10, value=1)

        # Toggle - Play
        self.play_toggle = Toggle(label='Play', button_type="primary")
        self.play_toggle.on_click(self.on_toggle_play)

        # Plot - EEG temporal signal
        self.plot_signal = figure(title='Temporal EEG signal',
                                  x_axis_label='Time [s]',
                                  y_axis_label='Amplitude',
                                  plot_height=450,
                                  plot_width=850)
        self.plot_signal.line(x='timestamps', y='values',
                              source=self.source)

        # Span - Event marker & label
        self.event_marker = Span(location=0, dimension='height',
                                 line_color='red', line_dash='dashed',
                                 line_width=3, visible=False)
        self.event_label = Label(x=0, y=0, y_units='screen',
                                 text_color='red', visible=False)
        self.plot_signal.renderers.extend([self.event_marker,
                                           self.event_label])

        # Create tab with layout
        column1 = column([self.select_pilot,
                          self.select_session,
                          row(self.select_labels),
                          self.slider_pre_event, self.slider_post_event,
                          self.checkbox_settings,
                          self.button_format, self.div_info])
        column2 = column([self.select_run, self.select_channel,
                          self.slider_t, self.slider_speed,
                          self.slider_win_len, self.play_toggle])
        column3 = column(self.plot_signal)
        layout = row(column1, column2, column3)
        return layout
