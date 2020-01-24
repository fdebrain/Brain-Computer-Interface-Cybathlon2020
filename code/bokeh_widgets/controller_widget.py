import sys
import glob
import logging
import numpy as np
import mne
from bokeh.io import curdoc
from bokeh.models.widgets import Div, Select, Slider, Toggle, Button
from bokeh.layouts import widgetbox, Spacer
from .observer import Observable

if sys.platform == 'win32':
    splitter = '\\'
else:
    splitter = '/'


class ControllerWidget(Observable):
    def __init__(self):
        Observable.__init__(self)
        self.data_path = '../Datasets/Pilots/Pilot_2'
        self.available_sessions = glob.glob(f'{self.data_path}/*')
        self.channel2idx = {}
        self._data = dict()
        self._signal_roi = dict(timestamps=[], values=[],
                                fs=None, channel_name=None)
        self.t = 0

    @property
    def fs(self):
        return self._data.get('fs', -1)

    @property
    def signal_roi(self):
        return self._signal_roi

    @signal_roi.setter
    def signal_roi(self, arg):
        for key in arg.keys():
            self.signal_roi[key] = arg[key]
        self._notify()

    def _notify(self):
        for obs in self._observers:
            obs.update(self._signal_roi)

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
        self.widget_title = Div(text='<b>Controller</b>',
                                align='center')

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

        layout = widgetbox([self.widget_title, self.select_session,
                            self.select_run, self.select_channel,
                            Spacer(height=5), self.win_len_slider,
                            Spacer(height=6), self.play_toggle])
        return layout

    def on_session_change(self, attr, old, new):
        logging.info(f'Select controller session {new}')
        available_runs = glob.glob(f'{self.data_path}/{new}/vhdr/*.vhdr')
        self.select_run.options = [''] + available_runs

    def on_run_change(self, attr, old, new):
        logging.info(f'Select formatter run {new}')
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

        # Extract current signal ROI
        start = int(self.t * self.fs)
        end = start + self.win_len
        ts = self._data['timestamps'][start:end]
        eeg = self._data['values'][self.channel_idx, start:end]
        self.signal_roi = dict(timestamps=ts, values=eeg, channel_name=new)

    def update_win_len(self, attr, old, new):
        logging.info(f'Win_len update: {new}')
        start = int(self.t * self.fs)
        end = start + new
        ts = self._data['timestamps'][start:end]
        eeg = self._data['values'][self.channel_idx, start:end]
        logging.info(f'Win_len update: {ts.shape} - {eeg.shape}')
        self.signal_roi = dict(timestamps=ts, values=eeg)

    # TODO: move to visualizer tab
    def callback_play(self):
        shift = 5
        start = int(self.t*self.fs + self.win_len)
        end = start + shift

        ts = np.roll(self._signal_roi['timestamps'], -shift)
        ts[-shift:] = self._data['timestamps'][start:end]
        eeg = np.roll(self._signal_roi['values'], -shift, axis=-1)
        eeg[-shift:] = self._data['values'][self.channel_idx, start:end]
        self.signal_roi = dict(timestamps=ts, values=eeg)
        self.t += shift / self.fs

    def on_toggle_play(self, state):
        if state:
            logging.info('Play')
            self.play_toggle.button_type = 'warning'
            self.callback = curdoc().add_periodic_callback(self.callback_play,
                                                           100)
        else:
            logging.info('Stop')
            self.play_toggle.button_type = 'primary'
            curdoc().remove_periodic_callback(self.callback)
