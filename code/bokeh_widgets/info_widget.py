import logging
from bokeh.models.widgets import Div
from bokeh.layouts import column
from .observer import Observer


class InfoWidget(Observer):
    def __init__(self):
        super(InfoWidget, self).__init__()
        self.div = Div(text='<h4>Run info:</h4 ><br>')

    def update(self, data):
        fs = data['fs']
        n_channels = data['values'].shape[0]
        recording_time = int((data['values'].shape[-1] / fs) / 60)

        self.div.text = '<h4>Run info:</h4>'
        self.div.text += f'<b>Frequency:</b> {fs} Hz<br>'
        self.div.text += f'<b>Channels:</b> {n_channels} <br>'
        self.div.text += f'<b>Recording time:</b> {recording_time} mn <br>'

    def create_widget(self):
        layout = column(self.div)
        return layout
