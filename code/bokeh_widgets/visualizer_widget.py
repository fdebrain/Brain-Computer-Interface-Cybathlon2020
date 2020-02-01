import sys
import glob
import logging
import numpy as np
import mne
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import Div, Select, Slider, Toggle
from bokeh.layouts import column, row

if sys.platform == 'win32':
    splitter = '\\'
else:
    splitter = '/'


class VisualizerWidget:
    def __init__(self):
        self.data_path = '../Datasets/Pilots/Pilot_2'
        self.available_sessions = glob.glob(f'{self.data_path}/*')
        self.channel2idx = {}
        self.t = 0
        self._data = dict()
        self.source = ColumnDataSource(data=dict(timestamps=[], values=[]))

    @property
    def fs(self):
        return self._data.get('fs', -1)

    @property
    def win_len(self):
        return self.win_len_slider.value

    @property
    def channel_name(self):
        return self.select_channel.value

    @property
    def channel_idx(self):
        return self.channel2idx[self.channel_name]

    def create_widget(self):
        # Select session
        self.select_session = Select(title="Session", options=[''])
        self.select_session.options += [session_path.split(splitter)[-1]
                                        for session_path in self.available_sessions]
        self.select_session.on_change('value', self.on_session_change)

        # Select run
        self.select_run = Select(title="Run")
        self.select_run.on_change('value', self.on_run_change)

        # Select channel
        self.select_channel = Select(title="Channel")
        self.select_channel.on_change('value', self.on_channel_change)

        # Slider Window length
        self.win_len_slider = Slider(title="Win length", start=100,
                                     end=1000, value=250)
        self.win_len_slider.on_change('value', self.update_win_len)

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

        self.div_info = Div()

        column1 = column(self.select_session,
                         self.select_run, self.select_channel,
                         self.win_len_slider, self.play_toggle)
        column2 = column(self.plot_signal)
        return row(column1, column2)

    def on_session_change(self, attr, old, new):
        logging.info(f'Select visualizer session {new}')
        available_runs = glob.glob(f'{self.data_path}/{new}/vhdr/*.vhdr')
        self.select_run.options = [''] + available_runs

    def on_run_change(self, attr, old, new):
        logging.info(f'Select visualizer run {new}')
        raw = mne.io.read_raw_brainvision(vhdr_fname=new,
                                          preload=False,
                                          verbose=False)
        available_channels = raw.ch_names
        self.select_channel.options = [''] + available_channels
        self.channel2idx = {c: i for i, c in enumerate(available_channels)}

        self._data['fs'] = raw.info['sfreq']
        self._data['values'], self._data['timestamps'] = raw.get_data(
            return_times=True)

    def on_channel_change(self, attr, old, new):
        logging.info(f'Select channel {new}')

        # Update source
        start = int(self.t * self.fs)
        end = start + self.win_len
        ts = self._data['timestamps'][start:end]
        eeg = self._data['values'][self.channel_idx, start:end]
        self.source.data = dict(timestamps=ts, values=eeg)

        # Update plot axis label
        self.plot_signal.yaxis.axis_label = f'Amplitude - {self.channel_name}'

    def update_win_len(self, attr, old, new):
        logging.info(f'Win_len update: {new}')
        start = int(self.t * self.fs)
        end = start + new
        ts = self._data['timestamps'][start:end]
        eeg = self._data['values'][self.channel_idx, start:end]
        logging.info(f'Win_len update: {ts.shape} - {eeg.shape}')
        self.source.data = dict(timestamps=ts, values=eeg)

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
