import logging

from bokeh.io import curdoc
from bokeh.models.widgets import Tabs, Panel
from bokeh_widgets.formatter_widget import FormatterWidget
from bokeh_widgets.visualizer_widget import VisualizerWidget
from bokeh_widgets.trainer_widget import TrainerWidget
from bokeh_widgets.test_widget import TestWidget
from bokeh_widgets.player_widget import PlayerWidget

document = curdoc()

# Setup logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Widgets and tabs
formatter = FormatterWidget().create_widget()
visualizer = VisualizerWidget().create_widget()
trainer = TrainerWidget().create_widget()
tester = TestWidget().create_widget()
player = PlayerWidget(parent=document).create_widget()


# Layout
tab_format = Panel(child=formatter, title='Formatter')
tab_visualize = Panel(child=visualizer, title='Visualizer')
tab_train = Panel(child=trainer, title='Trainer')
tab_test = Panel(child=tester, title='Tester')
tab_play = Panel(child=player, title='Player')
tabs = Tabs(tabs=[tab_format, tab_visualize, tab_train, tab_test, tab_play])
document.add_root(tabs)
