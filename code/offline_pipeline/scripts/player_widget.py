import logging
import glob
from bokeh.models.widgets import Div, Select, Button
from bokeh.layouts import widgetbox
from bokeh_widgets.observer import Observable, Observer


class PlayerWidget(Observable, Observer):
    def __init__(self):
        Observable.__init__(self)
        Observer.__init__(self)

        # Launch game button
        self.button_launch_game = Button(label='Launch Game',
                                         button_type='primary')

        # Autoplay toggle
        # Send events toggle

        # Game log file
        game_logs_path = '/media/fdebrain/Local Drive1/Desktop/ETH Courses/Spring 2019/master-thesis-cybathlon/game/log/raceLog*.txt'
        logfiles = glob.glob(game_logs_path)
        if len(logfiles) > 0:
            self.logfilename = logfiles[-1]
        else:
            logging.error('No log file available. Please launch the game')
            sys.exit(0)

    def on_launch_game(self):
        logging.info('Lauching Cybathlon game')
