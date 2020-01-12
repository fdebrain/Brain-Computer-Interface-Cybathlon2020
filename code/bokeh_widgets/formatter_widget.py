import logging
import glob
import numpy as np
from collections import Counter
import mne
from bokeh.models.widgets import Div, Select, Button
from bokeh.layouts import widgetbox
from formatting_functions.formatter import FormatterVHDR


class FormatterWidget:
    def __init__(self):
        self.data_path = '../Datasets/Pilots/Pilot_2'
        self.available_sessions = glob.glob(f'{self.data_path}/*')

    def create_widget(self):
        self.widget_title = Div(text='<b>Formatter</b>',
                                align='center', style={'color': '#000000'})
        self.select_session = Select(title="Session", options=[''])
        self.select_session.options += [session_path.split('/')[-1]
                                        for session_path in self.available_sessions]
        self.select_session.on_change('value', self.on_session_change)

        self.button_format = Button(label="Format", button_type="primary")
        self.button_format.on_click(self.on_format)

        self.info = Div()

        # Create tab with layout
        layout = widgetbox([self.widget_title, self.select_session,
                            self.button_format, self.info])
        return layout

    def on_session_change(self, attr, old, new):
        logging.info(f'Select formatter session {new}')
        available_runs = glob.glob(f'{self.data_path}/{new}/vhdr/*.vhdr')

        # Get session info
        fs = None
        events_counter = dict()
        n_channels, n_samples = None, None
        for run in available_runs:
            raw = mne.io.read_raw_brainvision(vhdr_fname=run,
                                              preload=False)

            fs = int(raw.info['sfreq'])
            n_channels = len(raw.ch_names)
            n_samples = raw.n_times
            events = Counter([e[-1]
                              for e in mne.events_from_annotations(raw)[0]])
            for event in events:
                events_counter[event] = events_counter.get(
                    event, 0) + events[event]

        # Update displayed info
        self.info.text = f'<b>Sampling frequency</b>: {fs} Hz <br>'
        self.info.text += f'<b>Nb of channels</b>: {n_channels} <br>'
        self.info.text += f'<b>Nb of samples</b>: {n_samples} '
        self.info.text += f'({int(n_samples/(fs*60))} mn) <br>'
        self.info.text += f'<b>Nb of occurence per events:</b><br>'
        for event, count in events_counter.items():
            if count > 5:
                self.info.text += f'&emsp; <b>{event}:</b>  {count} <br>'

        # Reset button state
        self.button_format.button_type = "primary"
        self.button_format.label = "Format"

    def on_format(self):
        assert self.button_format.label != "Formatted", 'Already formatted !'
        root = '../Datasets/Pilots/'
        pilot_idx = 2
        session_idx = self.select_session.value.split('_')[-1]
        labels_idx = [3, 4, 5, 6]

        ch_list = None
        remove_ch = ['Fp1', 'Fp2']
        if session_idx == '3':
            pre = -2.
            post = 8.
        else:
            pre = -5.
            post = 11.
        formatter = FormatterVHDR(root, root, pilot_idx, session_idx,
                                  labels_idx, ch_list, remove_ch,
                                  pre, post, mode='train', save=True,
                                  save_folder='formatted_filt_250Hz',
                                  preprocess=True, resample=True,
                                  control=False, save_as_trial=False)
        formatter.run()
        self.button_format.button_type = "success"
        self.button_format.label = "Formatted"
