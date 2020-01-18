import logging
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models.widgets import Div, Select
from bokeh.models import Panel, Button
from bokeh.models import ColumnDataSource
from .observer import Observer


class VisualizerTab(Observer):
    def __init__(self):
        super(VisualizerTab, self).__init__()
        self.signal = ColumnDataSource(data=dict(timestamps=[], values=[]))
        self.channel_name = None

    def update(self, signal):
        self.signal.data['timestamps'] = signal['timestamps']
        self.signal.data['values'] = signal['values']
        self.channel_name = signal['channel_name']
        self.plot.yaxis.axis_label = f'Amplitude - {self.channel_name}'

    def create_temporal_tab(self):
        self.plot = figure(title='Temporal EEG signal',
                           x_axis_label='Time [s]',
                           y_axis_label='Amplitude',
                           plot_height=450,
                           plot_width=850)

        self.plot.line(x='timestamps', y='values',
                       source=self.signal)
        layout = self.plot
        tab = Panel(child=layout, title='Temporal')
        return tab

    def create_spectral_tab(self):
        self.plot = figure(title='Spectral EEG signal',
                           x_axis_label='Time [s]',
                           y_axis_label='Frequency')

        self.plot.line(x='timestamps', y='values',
                       source=self.signal)
        layout = self.plot
        tab = Panel(child=layout, title='Spectral')
        return tab
