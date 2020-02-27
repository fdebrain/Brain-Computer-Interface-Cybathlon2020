import logging
from pathlib import Path

import numpy as np
import mne
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import Div, Select, Slider, Toggle, Span, Label
from bokeh.layouts import column, row


class VisualizerWidget:
    def __init__(self):
        self.data_path = Path('../Datasets/Pilots/Pilot_2')
        self.channel2idx = {}
        self._t = 0
        self._data = dict()
        self.source = ColumnDataSource(data=dict(timestamps=[], values=[]))
        self.source_events = ColumnDataSource(data=dict(events=[], actions=[]))

    @property
    def available_sessions(self):
        sessions = self.data_path.glob('*')
        return [''] + [s.name for s in sessions]

    @property
    def session_path(self):
        return self.data_path / self.select_session.value

    @property
    def available_runs(self):
        runs = self.session_path.glob('vhdr/*.vhdr')
        return [''] + [r.name for r in runs]

    @property
    def run_path(self):
        return self.session_path / 'vhdr' / self.select_run.value

    @property
    def fs(self):
        return self._data.get('fs', -1)

    @property
    def t_max(self):
        return len(self._data.get('timestamps', 0))

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

        # Extract ROI signal
        start = int(self.t * self.fs)
        end = start + self.win_len
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
    def channel_name(self):
        return self.select_channel.value

    @property
    def channel_idx(self):
        return self.channel2idx[self.channel_name]

    def on_session_change(self, attr, old, new):
        logging.info(f'Select visualizer session {new}')
        self.update_widgets()

    def on_run_change(self, attr, old, new):
        logging.info(f'Select visualizer run {new}')
        raw = mne.io.read_raw_brainvision(vhdr_fname=self.run_path,
                                          preload=True,
                                          verbose=False)

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
        # if action in [3, 4, 5, 6]]
        logging.info(self._data['events'])
        self.t = 0
        self.update_widgets()

    def update_widgets(self):
        self.select_session.options = self.available_sessions
        self.select_run.options = self.available_runs

    def on_channel_change(self, attr, old, new):
        logging.info(f'Select channel {new}')

        # Update source
        start = int(self._t * self.fs)
        end = start + self.win_len
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
        shift = 5
        start = int(self.t*self.fs + self.win_len)
        end = start + shift

        ts = np.roll(self.source.data['timestamps'], -shift)
        ts[-shift:] = self._data['timestamps'][start:end]
        eeg = np.roll(self.source.data['values'], -shift, axis=-1)
        eeg[-shift:] = self._data['values'][self.channel_idx, start:end]
        self.source.data = dict(timestamps=ts, values=eeg)
        self.t += shift / self.fs

    def on_toggle_play(self, active):
        assert self.channel_name != '', 'Select a channel first!'
        if active:
            logging.info('Play')
            self.play_toggle.button_type = 'warning'
            self.callback = curdoc().add_periodic_callback(self.callback_play,
                                                           100)
        else:
            logging.info('Stop')
            self.play_toggle.button_type = 'primary'
            curdoc().remove_periodic_callback(self.callback)

    def create_widget(self):
        # Select session
        self.select_session = Select(title="Session")
        self.select_session.options = self.available_sessions
        self.select_session.on_change('value', self.on_session_change)

        # Select run
        self.select_run = Select(title="Run")
        self.select_run.on_change('value', self.on_run_change)

        # Select channel
        self.select_channel = Select(title="Channel")
        self.select_channel.on_change('value', self.on_channel_change)

        # Slider - Navigate through signal
        self.slider_t = Slider(title="Navigate", start=0,
                                     end=1000, value=0)
        self.slider_t.on_change('value', self.on_t_change)

        # Slider - Window length
        self.slider_win_len = Slider(title="Win length", start=100,
                                     end=1000, value=250)
        self.slider_win_len.on_change('value', self.on_win_len_change)

        # Toggle play button
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
                                           self.event_label, ])

        # Div - Metadata
        self.div_info = Div()

        column1 = column(self.select_session,
                         self.select_run, self.select_channel,
                         self.slider_t,
                         self.slider_win_len, self.play_toggle)
        column2 = column(self.plot_signal)
        return row(column1, column2)
