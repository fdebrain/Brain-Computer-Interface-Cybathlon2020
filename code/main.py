import logging

from bokeh.io import curdoc
from bokeh.models.widgets import Tabs, Panel
from bokeh_widgets.formatter_widget import FormatterWidget
from bokeh_widgets.trainer_widget import TrainerWidget
from bokeh_widgets.test_widget import TestWidget
from bokeh_widgets.player_widget import PlayerWidget
from bokeh_widgets.warmup_widget import WarmUpWidget

document = curdoc()

# Setup logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Widgets and tabs
formatter = FormatterWidget().create_widget()
trainer = TrainerWidget().create_widget()
tester = TestWidget().create_widget()
warmup = WarmUpWidget(parent=document).create_widget()
player = PlayerWidget(parent=document).create_widget()

# Layout
tab_format = Panel(child=formatter, title='Format')
tab_train = Panel(child=trainer, title='Train')
tab_test = Panel(child=tester, title='Test')
tab_warmup = Panel(child=warmup, title='Warmup')
tab_play = Panel(child=player, title='Play')
tabs = Tabs(tabs=[tab_format, tab_train, tab_test, tab_warmup, tab_play])
document.add_root(tabs)
