# Formatting data in dataloader: progress bar
# Model widget (load data, select preprocessing, select model, train button)

import logging
from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.models.widgets import Tabs
from bokeh_widgets.formatter_widget import FormatterWidget
from bokeh_widgets.controller_widget import ControllerWidget
from bokeh_widgets.visualizer_tab import VisualizerTab
from bokeh_widgets.model_widget import ModelWidget
from bokeh_widgets.player_widget import PlayerWidget

# Setup logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Widgets and tabs
formatter = FormatterWidget()
visu_controller = ControllerWidget()
trainer = ModelWidget()
player = PlayerWidget()
tab1 = VisualizerTab()
tab2 = VisualizerTab()

visu_controller.attach(tab1)

# Layout
document = curdoc()
document.add_root(row(formatter.create_widget(),
                      visu_controller.create_widget(),
                      trainer.create_widget(),
                      player.create_widget()))
tabs = Tabs(tabs=[tab1.create_temporal_tab(),
                  tab2.create_spectral_tab()])
document.add_root(row(tabs))
