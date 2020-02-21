import sys
import logging
import glob
from collections import Counter
import mne
from bokeh.io import curdoc
from bokeh.models.widgets import Div, Select, Button, Slider, CheckboxButtonGroup
from bokeh.layouts import widgetbox, row
from formatting_functions.formatter import FormatterVHDR
from src.vhdr_formatter import format_session


if sys.platform == 'win32':
    splitter = '\\'
else:
    splitter = '/'


class FormatterWidget:
    def __init__(self):
        self.data_path = '../Datasets/Pilots/Pilot_2'
        self.available_sessions = glob.glob(f'{self.data_path}/*')

    @property
    def labels_idx(self):
        return [int(s.value) for s in self.select_labels]

    @property
    def session_idx(self):
        return self.select_session.value.split('_')[-1]

    @property
    def available_runs(self):
        return glob.glob(f'{self.data_path}/Session_{self.session_idx}/vhdr/*.vhdr')

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
    def should_resample(self):
        return 'Resample to 250Hz' in self.selected_settings

    @property
    def should_preprocess(self):
        return 'Preprocess' in self.selected_settings

    def on_extract_change(self, attr, old, new):
        logging.info(
            f'Epochs extracted ({self.pre},{self.post}) around marker')
        self.button_format.button_type = "primary"
        self.button_format.label = "Format"

    def on_session_change(self, attr, old, new):
        logging.info(f'Select session {new}')
        logging.info('For sessions <=3: MI starts 4s after marker')
        logging.info('For sessions > 3: MI starts 7s after marker')

        # Reset button state
        self.button_format.button_type = "primary"
        self.button_format.label = "Format"

        # Get session info
        fs = None
        events_counter = dict()
        n_channels, n_samples = 0, 0
        for run in self.available_runs:
            raw = mne.io.read_raw_brainvision(vhdr_fname=run,
                                              preload=False,
                                              verbose=False)

            fs = int(raw.info['sfreq'])
            n_channels = len(raw.ch_names)
            n_samples += raw.n_times
            events = Counter([e[-1]for e in mne.events_from_annotations(raw,
                                                                        verbose=False)[0]])
            for event in events:
                events_counter[event] = events_counter.get(event, 0) + \
                    events[event]

        # Update displayed info
        self.div_info.text = f'<b>Sampling frequency</b>: {fs} Hz <br>'
        self.div_info.text += f'<b>Nb of channels</b>: {n_channels} <br>'
        self.div_info.text += f'<b>Nb of samples</b>: {n_samples} '
        self.div_info.text += f'({int(n_samples/(fs*60))} mn) <br>'
        self.div_info.text += f'<b>Nb of occurence per events:</b><br>'
        for event, count in events_counter.items():
            if count > 5:
                self.div_info.text += f'&emsp; <b>{event}:</b>  {count} <br>'

    def on_format_start(self):
        assert self.select_session.value != '', \
            'Select a session to format first !'
        self.button_format.button_type = "warning"
        self.button_format.label = "Formatting..."
        curdoc().add_next_tick_callback(self.on_format)

    def on_format(self):
        session_path = f'{self.data_path}/Session_{self.session_idx}'
        remove_ch = ['Fp1', 'Fp2']
        extraction_settings = dict(pre=self.pre, post=self.post,
                                   marker_encodings=self.labels_idx)
        preprocess_settings = dict(resample=self.should_resample,
                                   preprocess=self.should_preprocess,
                                   remove_ch=remove_ch)
        save_path = f'{session_path}'

        try:
            format_session(self.available_runs,
                           save_path,
                           extraction_settings,
                           preprocess_settings)
        except Exception as e:
            logging.info(f'Failed to format - {e}')
            self.button_format.button_type = "danger"
            self.button_format.label = "Failed"
            return

        self.button_format.button_type = "success"
        self.button_format.label = "Formatted"

    def create_widget(self):
        self.select_session = Select(title="Session", options=[''])
        self.select_session.options += [session_path.split(splitter)[-1]
                                        for session_path in self.available_sessions]
        self.select_session.on_change('value', self.on_session_change)

        self.select_labels = [Select(title=f'Label {id+1}',
                                     options=[str(i) for i in range(30)],
                                     value=str(id+3),
                                     width=80)
                              for id in range(4)]

        self.slider_pre_event = Slider(start=-10, end=10, value=-5,
                                       title='Window start (s before event)')
        self.slider_pre_event.on_change('value', self.on_extract_change)
        self.slider_post_event = Slider(start=-10, end=15, value=11,
                                        title='Window end (s after event)')
        self.slider_post_event.on_change('value', self.on_extract_change)

        self.checkbox_settings = CheckboxButtonGroup(
            labels=['Resample to 250Hz', 'Preprocess'])

        self.button_format = Button(label="Format", button_type="primary")
        self.button_format.on_click(self.on_format_start)

        self.div_info = Div()

        # Create tab with layout
        layout = widgetbox([self.select_session, row(self.select_labels),
                            self.slider_pre_event, self.slider_post_event,
                            self.checkbox_settings,
                            self.button_format, self.div_info])
        return layout
