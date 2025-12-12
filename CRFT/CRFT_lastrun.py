#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on décembre 12, 2025, at 17:59
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'CRFT'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '00',
    'session': '001',
    'language': ["English", "French", "Spanish"],
    'handedness': ["Right", "Left"],
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\experiment\\Desktop\\CRFT\\CRFT_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('welcome_adv_key') is None:
        # initialise welcome_adv_key
        welcome_adv_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='welcome_adv_key',
        )
    if deviceManager.getDevice('intro_adv_resp') is None:
        # initialise intro_adv_resp
        intro_adv_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_adv_resp',
        )
    if deviceManager.getDevice('fb_test_adv_resp') is None:
        # initialise fb_test_adv_resp
        fb_test_adv_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='fb_test_adv_resp',
        )
    if deviceManager.getDevice('start_block_adv_key') is None:
        # initialise start_block_adv_key
        start_block_adv_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_block_adv_key',
        )
    if deviceManager.getDevice('inst_adv_resp') is None:
        # initialise inst_adv_resp
        inst_adv_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='inst_adv_resp',
        )
    if deviceManager.getDevice('check_fb_adv_resp') is None:
        # initialise check_fb_adv_resp
        check_fb_adv_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='check_fb_adv_resp',
        )
    if deviceManager.getDevice('reminder_adv_resp') is None:
        # initialise reminder_adv_resp
        reminder_adv_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='reminder_adv_resp',
        )
    if deviceManager.getDevice('setup_resp') is None:
        # initialise setup_resp
        setup_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='setup_resp',
        )
    if deviceManager.getDevice('space_resp') is None:
        # initialise space_resp
        space_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='space_resp',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('adv_fb_resp') is None:
        # initialise adv_fb_resp
        adv_fb_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='adv_fb_resp',
        )
    if deviceManager.getDevice('by_resp') is None:
        # initialise by_resp
        by_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='by_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    expShelf = data.shelf.Shelf(scope='experiment', expPath=_thisDir)
    # create uniform conditions for counterbalance
    counterbalanceConditions = []
    for n in range(2):
        counterbalanceConditions.append({
            'group': n,
            'probability': 1/2,
            'cap': 1
        })
    
    # create counterbalance object for counterbalance 
    counterbalance = data.Counterbalancer(
        shelf=expShelf,
        entry='counterbalance',
        conditions=counterbalanceConditions,
        nReps=80
    )
    
    # --- Initialize components for Routine "exp_settings" ---
    # Run 'Begin Experiment' code from experiment_settings_code
    n_blocks = 4
    current_block = 0
    trial_n = 0
    n_targets = 5  # number of targets on screen
    n_IDs = 5  # 5 levels of Index of Difficulty
    practice_reps = 1 # 1 round of slightly different IDs
    reps_per_block = 1  # start at 1, this updates later depending on block type
    main_blocks_reps = 4  # change to 4 for the actual task!!
    show_inst = 1
    trials_block = 0
    key_presses = 5*2 + 1 # 11!
    inst_file = ""
    practice_taps = 30
    
    # Run 'Begin Experiment' code from vars_to_process
    space_durations = []
    target_diameters = []
    index_difficulties = []
    movement_times = []
    
    # --- Initialize components for Routine "language_settings" ---
    # Run 'Begin Experiment' code from language_localiser_code
    import pandas as pd
    # make sure lang_code is defined and set to EN as default
    lang_code = "EN"
    # read excel file with message according to language codes
    messages_df = pd.read_excel('messages.xlsx')
    # create a global dictionary
    MESSAGES = {}
    # assign each value of language to the corresponding key (language code)
    for idx, row in messages_df.iterrows():
        key = row['message']
        MESSAGES[key] = {}
        for col in row.index:
            if col != 'message':
                MESSAGES[key][col] = row[col]
    # create global variables with the list of messages to be usable throuhgout the experiment
    for key in MESSAGES:
        globals()[key] = MESSAGES[key].get(lang_code, MESSAGES[key]['EN'])  # fallback to English if language is not localised
    
    # --- Initialize components for Routine "welcome" ---
    order_text = visual.TextStim(win=win, name='order_text',
        text='',
        font='Arial',
        pos=(0.8, 0.45), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    task_text = visual.TextStim(win=win, name='task_text',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    stylus_text = visual.TextStim(win=win, name='stylus_text',
        text='',
        font='Arial',
        pos=(0, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    adv_text = visual.TextStim(win=win, name='adv_text',
        text='',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.045, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    welcome_adv_key = keyboard.Keyboard(deviceName='welcome_adv_key')
    
    # --- Initialize components for Routine "introduction" ---
    intro_text1 = visual.TextStim(win=win, name='intro_text1',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    stylus_image = visual.ImageStim(
        win=win,
        name='stylus_image', 
        image='images/stylus_exe.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    intro_text2 = visual.TextStim(win=win, name='intro_text2',
        text='',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    adv_text_2 = visual.TextStim(win=win, name='adv_text_2',
        text='',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.045, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    intro_adv_resp = keyboard.Keyboard(deviceName='intro_adv_resp')
    
    # --- Initialize components for Routine "test_screen" ---
    # Run 'Begin Experiment' code from test_screen_code
    import numpy as np
    n = 30
    # last value is number
    circle_sizes = np.linspace(0.02, 0.2, n) 
    circle_xs = np.linspace(-0.4, 0.4, n)
    circle_ys = np.linspace(-0.3, 0.3, n)
    circle_size = 0
    circle_x = 0
    circle_y = 0
    circle_color = 'grey'
    correct_taps = 0
    this_tap = 0
    test_text = visual.TextStim(win=win, name='test_text',
        text='',
        font='Arial',
        pos=(0, 0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    circle_test = visual.ShapeStim(
        win=win, name='circle_test',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    mouse_test = event.Mouse(win=win)
    x, y = [None, None]
    mouse_test.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "fb_test" ---
    test_screen_fb_text = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.04,
         size=(1, 0.5), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='test_screen_fb_text',
         depth=0, autoLog=True,
    )
    fb_test_adv_text = visual.TextStim(win=win, name='fb_test_adv_text',
        text='',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    fb_test_adv_resp = keyboard.Keyboard(deviceName='fb_test_adv_resp')
    
    # --- Initialize components for Routine "start_block" ---
    block_n_text = visual.TextStim(win=win, name='block_n_text',
        text='',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.045, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 0.4980, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    start_block_text = visual.TextStim(win=win, name='start_block_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    start_block_adv_text = visual.TextStim(win=win, name='start_block_adv_text',
        text='',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.045, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    start_block_adv_key = keyboard.Keyboard(deviceName='start_block_adv_key')
    
    # --- Initialize components for Routine "instructions" ---
    # Run 'Begin Experiment' code from inst_code
    adv_time = 0.5
    inst_text = visual.TextStim(win=win, name='inst_text',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.0275, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    inst_adv_resp = keyboard.Keyboard(deviceName='inst_adv_resp')
    inst_image = visual.ImageStim(
        win=win,
        name='inst_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    inst_adv_text = visual.TextStim(win=win, name='inst_adv_text',
        text='',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    animation_video = visual.MovieStim(
        win, name='animation_video',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=True,
        pos=(0, 0), size=(0.8, 0.7), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=-5
    )
    
    # --- Initialize components for Routine "comp_check" ---
    # Run 'Begin Experiment' code from comp_check_code
    y_locs = [0.3, 0.2, 0.1, 0, -0.1, -0.2]
    comp_check_text = visual.TextStim(win=win, name='comp_check_text',
        text='',
        font='Arial',
        pos=(0, 0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    buttonA = visual.ButtonStim(win, 
        text='x', font='Arial',
        pos=[0,0],
        letterHeight=0.03,
        size=(0.05, 0.05), 
        ori=0.0
        ,borderWidth=0.5,
        fillColor='white', borderColor=[-1.0000, -1.0000, -1.0000],
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='buttonA',
        depth=-2
    )
    buttonA.buttonClock = core.Clock()
    textboxA = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=[0,0], draggable=False,      letterHeight=0.025,
         size=(1, 0.1), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center-left',
         anchor='center-left', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textboxA',
         depth=-3, autoLog=True,
    )
    buttonB = visual.ButtonStim(win, 
        text='x', font='Arial',
        pos=[0,0],
        letterHeight=0.03,
        size=(0.05, 0.05), 
        ori=0.0
        ,borderWidth=0.5,
        fillColor='white', borderColor=[-1.0000, -1.0000, -1.0000],
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='buttonB',
        depth=-4
    )
    buttonB.buttonClock = core.Clock()
    textboxB = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=[0,0], draggable=False,      letterHeight=0.025,
         size=(1, 0.1), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center-left',
         anchor='center-left', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textboxB',
         depth=-5, autoLog=True,
    )
    buttonC = visual.ButtonStim(win, 
        text='x', font='Arial',
        pos=[0,0],
        letterHeight=0.03,
        size=(0.05, 0.05), 
        ori=0.0
        ,borderWidth=0.5,
        fillColor='white', borderColor=[-1.0000, -1.0000, -1.0000],
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='buttonC',
        depth=-6
    )
    buttonC.buttonClock = core.Clock()
    textboxC = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=[0,0], draggable=False,      letterHeight=0.025,
         size=(1, 0.1), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center-left',
         anchor='center-left', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textboxC',
         depth=-7, autoLog=True,
    )
    buttonD = visual.ButtonStim(win, 
        text='x', font='Arial',
        pos=[0,0],
        letterHeight=0.03,
        size=(0.05, 0.05), 
        ori=0.0
        ,borderWidth=0.5,
        fillColor='white', borderColor=[-1.0000, -1.0000, -1.0000],
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='buttonD',
        depth=-8
    )
    buttonD.buttonClock = core.Clock()
    textboxD = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=[0,0], draggable=False,      letterHeight=0.025,
         size=(1, 0.1), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center-left',
         anchor='center-left', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textboxD',
         depth=-9, autoLog=True,
    )
    buttonE = visual.ButtonStim(win, 
        text='x', font='Arial',
        pos=[0,0],
        letterHeight=0.03,
        size=(0.05, 0.05), 
        ori=0.0
        ,borderWidth=0.5,
        fillColor='white', borderColor=[-1.0000, -1.0000, -1.0000],
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='buttonE',
        depth=-10
    )
    buttonE.buttonClock = core.Clock()
    textboxE = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=[0,0], draggable=False,      letterHeight=0.025,
         size=(1, 0.1), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center-left',
         anchor='center-left', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textboxE',
         depth=-11, autoLog=True,
    )
    buttonF = visual.ButtonStim(win, 
        text='x', font='Arial',
        pos=[0,0],
        letterHeight=0.03,
        size=(0.05, 0.05), 
        ori=0.0
        ,borderWidth=0.5,
        fillColor='white', borderColor=[-1.0000, -1.0000, -1.0000],
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='buttonF',
        depth=-12
    )
    buttonF.buttonClock = core.Clock()
    textboxF = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=[0,0], draggable=False,      letterHeight=0.025,
         size=(1, 0.1), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center-left',
         anchor='center-left', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textboxF',
         depth=-13, autoLog=True,
    )
    submit_button = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(0, -0.4),
        letterHeight=0.03,
        size=(0.2, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[0.6549, 0.6549, 0.6549], borderColor=None,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='submit_button',
        depth=-14
    )
    submit_button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "check_fb" ---
    corr_incorr_fb = visual.TextStim(win=win, name='corr_incorr_fb',
        text='',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    check_fb_text = visual.TextStim(win=win, name='check_fb_text',
        text='',
        font='Arial',
        pos=(0, -0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    check_fb_adv_text = visual.TextStim(win=win, name='check_fb_adv_text',
        text='',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    check_fb_adv_resp = keyboard.Keyboard(deviceName='check_fb_adv_resp')
    
    # --- Initialize components for Routine "reminder" ---
    reminder_title = visual.TextStim(win=win, name='reminder_title',
        text='',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 0.4980, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    reminder_text = visual.TextStim(win=win, name='reminder_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    reminder_adv_text = visual.TextStim(win=win, name='reminder_adv_text',
        text='',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    reminder_adv_resp = keyboard.Keyboard(deviceName='reminder_adv_resp')
    
    # --- Initialize components for Routine "setup" ---
    # Run 'Begin Experiment' code from setup_code
    import numpy as np
    # General parameters
    n_targets = 5
    start_x = 0.0
    start_y = -0.4
    start_diameter = 0.09
    angles_deg = [-60, -30, 0, 30, 60]
    angles_rad = [np.deg2rad(a) for a in angles_deg]
    distance = 0.424 # calculated from Carla's 458 pixels
    
    # Define the diameters
    target_diameters = [0.01176, 0.02353, 0.05, 0.1, 0.20588]
    thisExp.addData("target_diameters", target_diameters)
    # Calculate Index of Difficulty
    IDs = []
    for target_diameter in target_diameters:
        target_radius = target_diameter / 2
        start_radius = start_diameter / 2
        A = distance + start_radius + target_radius
        ID = np.log2((2 * A) / target_diameter)
        ID = round(ID, 2)
        IDs.append(ID)
    print(IDs)
    press_space_text = visual.TextStim(win=win, name='press_space_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    start_flag = visual.ShapeStim(
        win=win, name='start_flag',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, 0.2941, -1.0000], fillColor=[1.0000, 0.2941, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    flag_image = visual.ImageStim(
        win=win,
        name='flag_image', 
        image='images/flag.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    setup_resp = keyboard.Keyboard(deviceName='setup_resp')
    
    # --- Initialize components for Routine "countdown" ---
    start_count = visual.ShapeStim(
        win=win, name='start_count',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, -0.2235, -0.4431], fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=0.0, interpolate=True)
    target_1_count = visual.ShapeStim(
        win=win, name='target_1_count',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    target_2_count = visual.ShapeStim(
        win=win, name='target_2_count',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    target_3_count = visual.ShapeStim(
        win=win, name='target_3_count',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    target_4_count = visual.ShapeStim(
        win=win, name='target_4_count',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    target_5_count = visual.ShapeStim(
        win=win, name='target_5_count',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    rect_countdown = visual.Rect(
        win=win, name='rect_countdown',
        width=(1.3, 0.75)[0], height=(1.3, 0.75)[1],
        ori=0.0, pos=(0, -0.08), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=0.4, depth=-6.0, interpolate=True)
    countdown_3 = visual.TextStim(win=win, name='countdown_3',
        text='3',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    countdown_2 = visual.TextStim(win=win, name='countdown_2',
        text='2',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    countdown_1 = visual.TextStim(win=win, name='countdown_1',
        text='1',
        font='Arial',
        pos=(0, -0.2), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "trial" ---
    # Run 'Begin Experiment' code from trial_code
    cursor_size = 0.000
    # Initialize colors
    cursor_color = 'black'
    target_1_color = target_2_color = target_3_color = target_4_color = target_5_color = 'grey'
    start = visual.ShapeStim(
        win=win, name='start',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, -0.2235, -0.4431], fillColor=[1.0000, -0.4588, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    target_1 = visual.ShapeStim(
        win=win, name='target_1',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    target_2 = visual.ShapeStim(
        win=win, name='target_2',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    target_3 = visual.ShapeStim(
        win=win, name='target_3',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    target_4 = visual.ShapeStim(
        win=win, name='target_4',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    target_5 = visual.ShapeStim(
        win=win, name='target_5',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    cursor = visual.ShapeStim(
        win=win, name='cursor', vertices='star7',
        size=[1.0, 1.0],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    space_resp = keyboard.Keyboard(deviceName='space_resp')
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "blank" ---
    
    # --- Initialize components for Routine "fb" ---
    # Run 'Begin Experiment' code from feedback_code
    fb_time_x = 0
    fb_time_y = 0
    fb_taps_x = 0
    fb_taps_y = 0
    trial_text = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0.2), draggable=False,      letterHeight=0.07,
         size=(0.5, 0.5), borderWidth=2.0,
         color=[-1.0000, 0.4980, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trial_text',
         depth=-1, autoLog=True,
    )
    fb_text_time = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=[0,0], draggable=False,      letterHeight=0.04,
         size=(0.5, 0.5), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='fb_text_time',
         depth=-2, autoLog=True,
    )
    fb_text_taps = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=[0,0], draggable=False,      letterHeight=0.04,
         size=(0.5, 0.5), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='fb_text_taps',
         depth=-3, autoLog=True,
    )
    fb_adv_text = visual.TextStim(win=win, name='fb_adv_text',
        text='',
        font='Arial',
        pos=(0, -0.45), draggable=False, height=0.045, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    adv_fb_resp = keyboard.Keyboard(deviceName='adv_fb_resp')
    
    # --- Initialize components for Routine "controller" ---
    # Run 'Begin Experiment' code from controller_code
    cur_row = 0
    button_pressed = "none"
    show_self_assess = 1
    max_slides = 13   # length of rows in Excel - 1!!!
    
    # --- Initialize components for Routine "self_assess" ---
    imagery_icon = visual.ImageStim(
        win=win,
        name='imagery_icon', 
        image='images/imagery_icon.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.4), draggable=False, size=(0.12, 0.12),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    self_assess_text = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0.18), draggable=False,      letterHeight=0.03,
         size=(0.9, 0.4), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='self_assess_text',
         depth=-2, autoLog=True,
    )
    slider = visual.Slider(win=win, name='slider',
        startValue=None, size=(1.0, 0.05), pos=(0, -0.05), units=win.units,
        labels=(0,'','','','','','','','','',10), ticks=(0,1,2,3,4,5,6,7,8,9,10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[-1.0000, -1.0000, -1.0000], markerColor=[-1.0000, 0.4980, 1.0000], lineColor=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
        font='Arial', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    label_0 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(-0.5, -0.35), draggable=False,      letterHeight=0.03,
         size=(0.4, 0.3), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.8, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='bottom-center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='label_0',
         depth=-4, autoLog=True,
    )
    label_10 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0.5, -0.35), draggable=False,      letterHeight=0.03,
         size=(0.4, 0.3), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.8, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='bottom-center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='label_10',
         depth=-5, autoLog=True,
    )
    next_button = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(0.45, -0.4),
        letterHeight=0.03,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[0.6549, 0.6549, 0.6549], borderColor=[-1.0000, -1.0000, -1.0000],
        color=[0.2941, -0.6706, -0.6706], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='next_button',
        depth=-6
    )
    next_button.buttonClock = core.Clock()
    back_button = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(-0.45, -0.4),
        letterHeight=0.03,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor=[0.6549, 0.6549, 0.6549], borderColor=[-1.0000, -1.0000, -1.0000],
        color=[0.2941, -0.6706, -0.6706], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='back_button',
        depth=-7
    )
    back_button.buttonClock = core.Clock()
    double_tap_text = visual.TextStim(win=win, name='double_tap_text',
        text='',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "processing" ---
    
    # --- Initialize components for Routine "bye" ---
    # Run 'Begin Experiment' code from bye_code
    regression_plot = ""
    bye_text = visual.TextStim(win=win, name='bye_text',
        text='',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    regression_image = visual.ImageStim(
        win=win,
        name='regression_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    by_resp = keyboard.Keyboard(deviceName='by_resp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    # get group from shelf
    counterbalance.allocateGroup()
    # if slots and repeats are fully depleted, end the experiment now
    if counterbalance.finished:
        # first print and log a message to make it clear why the experiment ended
        print('Slots for Counterbalancer counterbalance have been fully depleted, ending experiment.')
        logging.exp('Slots for Counterbalancer counterbalance have been fully depleted, ending experiment.')
        endExperiment(thisExp, win=win)
    thisExp.addData('counterbalance.group', counterbalance.group)
    for _key, _val in counterbalance.params.items():
        thisExp.addData(f'counterbalance.{_key}', _val)
    thisExp.addData('counterbalance.remaining', counterbalance.remaining)
    thisExp.nextEntry()
    # the Routine "counterbalance" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "exp_settings" ---
    # create an object to store info about Routine exp_settings
    exp_settings = data.Routine(
        name='exp_settings',
        components=[],
    )
    exp_settings.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from experiment_settings_code
    # add group to output and variable to use for order
    expInfo['order'] = counterbalance.group
    order = expInfo['order']
    
    # store start times for exp_settings
    exp_settings.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    exp_settings.tStart = globalClock.getTime(format='float')
    exp_settings.status = STARTED
    exp_settings.maxDuration = None
    # keep track of which components have finished
    exp_settingsComponents = exp_settings.components
    for thisComponent in exp_settings.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "exp_settings" ---
    exp_settings.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=exp_settings,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            exp_settings.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in exp_settings.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "exp_settings" ---
    for thisComponent in exp_settings.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for exp_settings
    exp_settings.tStop = globalClock.getTime(format='float')
    exp_settings.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "exp_settings" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    language_loop = data.TrialHandler2(
        name='language_loop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('language_localiser.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(language_loop)  # add the loop to the experiment
    thisLanguage_loop = language_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLanguage_loop.rgb)
    if thisLanguage_loop != None:
        for paramName in thisLanguage_loop:
            globals()[paramName] = thisLanguage_loop[paramName]
    
    for thisLanguage_loop in language_loop:
        language_loop.status = STARTED
        if hasattr(thisLanguage_loop, 'status'):
            thisLanguage_loop.status = STARTED
        currentLoop = language_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisLanguage_loop.rgb)
        if thisLanguage_loop != None:
            for paramName in thisLanguage_loop:
                globals()[paramName] = thisLanguage_loop[paramName]
        
        # --- Prepare to start Routine "language_settings" ---
        # create an object to store info about Routine language_settings
        language_settings = data.Routine(
            name='language_settings',
            components=[],
        )
        language_settings.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from language_localiser_code
        if language == expInfo['language']:
            lang_code = ISO_code
            thisExp.addData("language_code", lang_code)
            # update global variables with new language
            for key in MESSAGES:
                globals()[key] = MESSAGES[key].get(lang_code, MESSAGES[key]['EN'])
        
        
        # store start times for language_settings
        language_settings.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        language_settings.tStart = globalClock.getTime(format='float')
        language_settings.status = STARTED
        language_settings.maxDuration = None
        # keep track of which components have finished
        language_settingsComponents = language_settings.components
        for thisComponent in language_settings.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "language_settings" ---
        language_settings.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisLanguage_loop, 'status') and thisLanguage_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=language_settings,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                language_settings.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in language_settings.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "language_settings" ---
        for thisComponent in language_settings.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for language_settings
        language_settings.tStop = globalClock.getTime(format='float')
        language_settings.tStopRefresh = tThisFlipGlobal
        # the Routine "language_settings" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisLanguage_loop as finished
        if hasattr(thisLanguage_loop, 'status'):
            thisLanguage_loop.status = FINISHED
        # if awaiting a pause, pause now
        if language_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            language_loop.status = STARTED
    # completed 1.0 repeats of 'language_loop'
    language_loop.status = FINISHED
    
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[order_text, task_text, stylus_text, adv_text, welcome_adv_key],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    order_text.setText('Order: ' + str(order))
    task_text.setText(welcome_msg)
    stylus_text.setColor([-1.0000, -1.0000, -1.0000], colorSpace='rgb')
    stylus_text.setText(stylus_msg)
    adv_text.setText(adv_msg)
    # create starting attributes for welcome_adv_key
    welcome_adv_key.keys = []
    welcome_adv_key.rt = []
    _welcome_adv_key_allKeys = []
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *order_text* updates
        
        # if order_text is starting this frame...
        if order_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            order_text.frameNStart = frameN  # exact frame index
            order_text.tStart = t  # local t and not account for scr refresh
            order_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(order_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            order_text.status = STARTED
            order_text.setAutoDraw(True)
        
        # if order_text is active this frame...
        if order_text.status == STARTED:
            # update params
            pass
        
        # *task_text* updates
        
        # if task_text is starting this frame...
        if task_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            task_text.frameNStart = frameN  # exact frame index
            task_text.tStart = t  # local t and not account for scr refresh
            task_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(task_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            task_text.status = STARTED
            task_text.setAutoDraw(True)
        
        # if task_text is active this frame...
        if task_text.status == STARTED:
            # update params
            pass
        
        # *stylus_text* updates
        
        # if stylus_text is starting this frame...
        if stylus_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            stylus_text.frameNStart = frameN  # exact frame index
            stylus_text.tStart = t  # local t and not account for scr refresh
            stylus_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stylus_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'stylus_text.started')
            # update status
            stylus_text.status = STARTED
            stylus_text.setAutoDraw(True)
        
        # if stylus_text is active this frame...
        if stylus_text.status == STARTED:
            # update params
            pass
        
        # *adv_text* updates
        
        # if adv_text is starting this frame...
        if adv_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            adv_text.frameNStart = frameN  # exact frame index
            adv_text.tStart = t  # local t and not account for scr refresh
            adv_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(adv_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            adv_text.status = STARTED
            adv_text.setAutoDraw(True)
        
        # if adv_text is active this frame...
        if adv_text.status == STARTED:
            # update params
            pass
        
        # *welcome_adv_key* updates
        waitOnFlip = False
        
        # if welcome_adv_key is starting this frame...
        if welcome_adv_key.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            # keep track of start time/frame for later
            welcome_adv_key.frameNStart = frameN  # exact frame index
            welcome_adv_key.tStart = t  # local t and not account for scr refresh
            welcome_adv_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_adv_key, 'tStartRefresh')  # time at next scr refresh
            # update status
            welcome_adv_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(welcome_adv_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(welcome_adv_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if welcome_adv_key.status == STARTED and not waitOnFlip:
            theseKeys = welcome_adv_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _welcome_adv_key_allKeys.extend(theseKeys)
            if len(_welcome_adv_key_allKeys):
                welcome_adv_key.keys = _welcome_adv_key_allKeys[-1].name  # just the last key pressed
                welcome_adv_key.rt = _welcome_adv_key_allKeys[-1].rt
                welcome_adv_key.duration = _welcome_adv_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=welcome,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "introduction" ---
    # create an object to store info about Routine introduction
    introduction = data.Routine(
        name='introduction',
        components=[intro_text1, stylus_image, intro_text2, adv_text_2, intro_adv_resp],
    )
    introduction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    intro_text1.setColor([-1.0000, -1.0000, -1.0000], colorSpace='rgb')
    intro_text1.setText(intro_msg)
    intro_text2.setColor([-1.0000, -1.0000, -1.0000], colorSpace='rgb')
    intro_text2.setText(sat_msg)
    adv_text_2.setText(adv_msg)
    # create starting attributes for intro_adv_resp
    intro_adv_resp.keys = []
    intro_adv_resp.rt = []
    _intro_adv_resp_allKeys = []
    # store start times for introduction
    introduction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    introduction.tStart = globalClock.getTime(format='float')
    introduction.status = STARTED
    thisExp.addData('introduction.started', introduction.tStart)
    introduction.maxDuration = None
    # keep track of which components have finished
    introductionComponents = introduction.components
    for thisComponent in introduction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "introduction" ---
    introduction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_text1* updates
        
        # if intro_text1 is starting this frame...
        if intro_text1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_text1.frameNStart = frameN  # exact frame index
            intro_text1.tStart = t  # local t and not account for scr refresh
            intro_text1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_text1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_text1.started')
            # update status
            intro_text1.status = STARTED
            intro_text1.setAutoDraw(True)
        
        # if intro_text1 is active this frame...
        if intro_text1.status == STARTED:
            # update params
            pass
        
        # *stylus_image* updates
        
        # if stylus_image is starting this frame...
        if stylus_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            stylus_image.frameNStart = frameN  # exact frame index
            stylus_image.tStart = t  # local t and not account for scr refresh
            stylus_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stylus_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'stylus_image.started')
            # update status
            stylus_image.status = STARTED
            stylus_image.setAutoDraw(True)
        
        # if stylus_image is active this frame...
        if stylus_image.status == STARTED:
            # update params
            pass
        
        # *intro_text2* updates
        
        # if intro_text2 is starting this frame...
        if intro_text2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_text2.frameNStart = frameN  # exact frame index
            intro_text2.tStart = t  # local t and not account for scr refresh
            intro_text2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_text2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_text2.started')
            # update status
            intro_text2.status = STARTED
            intro_text2.setAutoDraw(True)
        
        # if intro_text2 is active this frame...
        if intro_text2.status == STARTED:
            # update params
            pass
        
        # *adv_text_2* updates
        
        # if adv_text_2 is starting this frame...
        if adv_text_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            adv_text_2.frameNStart = frameN  # exact frame index
            adv_text_2.tStart = t  # local t and not account for scr refresh
            adv_text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(adv_text_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            adv_text_2.status = STARTED
            adv_text_2.setAutoDraw(True)
        
        # if adv_text_2 is active this frame...
        if adv_text_2.status == STARTED:
            # update params
            pass
        
        # *intro_adv_resp* updates
        waitOnFlip = False
        
        # if intro_adv_resp is starting this frame...
        if intro_adv_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_adv_resp.frameNStart = frameN  # exact frame index
            intro_adv_resp.tStart = t  # local t and not account for scr refresh
            intro_adv_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_adv_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_adv_resp.started')
            # update status
            intro_adv_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_adv_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_adv_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_adv_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_adv_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_adv_resp_allKeys.extend(theseKeys)
            if len(_intro_adv_resp_allKeys):
                intro_adv_resp.keys = _intro_adv_resp_allKeys[-1].name  # just the last key pressed
                intro_adv_resp.rt = _intro_adv_resp_allKeys[-1].rt
                intro_adv_resp.duration = _intro_adv_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=introduction,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            introduction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in introduction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "introduction" ---
    for thisComponent in introduction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for introduction
    introduction.tStop = globalClock.getTime(format='float')
    introduction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('introduction.stopped', introduction.tStop)
    # check responses
    if intro_adv_resp.keys in ['', [], None]:  # No response was made
        intro_adv_resp.keys = None
    thisExp.addData('intro_adv_resp.keys',intro_adv_resp.keys)
    if intro_adv_resp.keys != None:  # we had a response
        thisExp.addData('intro_adv_resp.rt', intro_adv_resp.rt)
        thisExp.addData('intro_adv_resp.duration', intro_adv_resp.duration)
    thisExp.nextEntry()
    # the Routine "introduction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    test_loop = data.TrialHandler2(
        name='test_loop',
        nReps=practice_taps, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(test_loop)  # add the loop to the experiment
    thisTest_loop = test_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTest_loop.rgb)
    if thisTest_loop != None:
        for paramName in thisTest_loop:
            globals()[paramName] = thisTest_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTest_loop in test_loop:
        test_loop.status = STARTED
        if hasattr(thisTest_loop, 'status'):
            thisTest_loop.status = STARTED
        currentLoop = test_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTest_loop.rgb)
        if thisTest_loop != None:
            for paramName in thisTest_loop:
                globals()[paramName] = thisTest_loop[paramName]
        
        # --- Prepare to start Routine "test_screen" ---
        # create an object to store info about Routine test_screen
        test_screen = data.Routine(
            name='test_screen',
            components=[test_text, circle_test, mouse_test],
        )
        test_screen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from test_screen_code
        # choose randomly size and position
        circle_size = np.random.choice(circle_sizes)
        circle_x = np.random.choice(circle_xs)
        circle_y = np.random.choice(circle_ys)
        
        #mouse.setPos([-0.5, -0.5])
        last_pos = mouse_test.getPos()
        last_time = core.getTime()
        this_tap = 0
        tap_detected = False
        
        if test_loop.thisN == 0:
            circle_color = [0.0902, -1.0000, -1.0000]
        else:
            circle_color = 'grey'
        test_text.setText(test_msg)
        circle_test.setFillColor(circle_color)
        circle_test.setPos((circle_x, circle_y))
        circle_test.setSize((circle_size, circle_size))
        circle_test.setLineColor(circle_color)
        # setup some python lists for storing info about the mouse_test
        mouse_test.x = []
        mouse_test.y = []
        mouse_test.leftButton = []
        mouse_test.midButton = []
        mouse_test.rightButton = []
        mouse_test.time = []
        mouse_test.corr = []
        mouse_test.clicked_name = []
        gotValidClick = False  # until a click is received
        # store start times for test_screen
        test_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        test_screen.tStart = globalClock.getTime(format='float')
        test_screen.status = STARTED
        thisExp.addData('test_screen.started', test_screen.tStart)
        test_screen.maxDuration = None
        # keep track of which components have finished
        test_screenComponents = test_screen.components
        for thisComponent in test_screen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "test_screen" ---
        test_screen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTest_loop, 'status') and thisTest_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from test_screen_code
            # Get current mouse position
            new_pos = mouse_test.getPos()
            
            # Detect a "tap" when the mouse moves and enough time has passed
            if not tap_detected and not np.array_equal(new_pos, last_pos):
                tap_detected = True
                if circle_test.contains(new_pos):
                    this_tap = 1
                else:
                    this_tap = 0
                continueRoutine = False
            
            last_pos = new_pos
            
            # *test_text* updates
            
            # if test_text is starting this frame...
            if test_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                test_text.frameNStart = frameN  # exact frame index
                test_text.tStart = t  # local t and not account for scr refresh
                test_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_text.started')
                # update status
                test_text.status = STARTED
                test_text.setAutoDraw(True)
            
            # if test_text is active this frame...
            if test_text.status == STARTED:
                # update params
                pass
            
            # *circle_test* updates
            
            # if circle_test is starting this frame...
            if circle_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                circle_test.frameNStart = frameN  # exact frame index
                circle_test.tStart = t  # local t and not account for scr refresh
                circle_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circle_test.started')
                # update status
                circle_test.status = STARTED
                circle_test.setAutoDraw(True)
            
            # if circle_test is active this frame...
            if circle_test.status == STARTED:
                # update params
                pass
            # *mouse_test* updates
            
            # if mouse_test is starting this frame...
            if mouse_test.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                mouse_test.frameNStart = frameN  # exact frame index
                mouse_test.tStart = t  # local t and not account for scr refresh
                mouse_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_test.started', t)
                # update status
                mouse_test.status = STARTED
                mouse_test.mouseClock.reset()
                prevButtonState = mouse_test.getPressed()  # if button is down already this ISN'T a new click
            if mouse_test.status == STARTED:  # only update if started and not finished!
                buttons = mouse_test.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(circle_test, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse_test):
                                gotValidClick = True
                                mouse_test.clicked_name.append(obj.name)
                        if not gotValidClick:
                            mouse_test.clicked_name.append(None)
                        # check whether click was in correct object
                        if gotValidClick:
                            _corr = 0
                            _corrAns = environmenttools.getFromNames(circle_test, namespace=locals())
                            for obj in _corrAns:
                                # is this object clicked on?
                                if obj.contains(mouse_test):
                                    _corr = 1
                            mouse_test.corr.append(_corr)
                        x, y = mouse_test.getPos()
                        mouse_test.x.append(x)
                        mouse_test.y.append(y)
                        buttons = mouse_test.getPressed()
                        mouse_test.leftButton.append(buttons[0])
                        mouse_test.midButton.append(buttons[1])
                        mouse_test.rightButton.append(buttons[2])
                        mouse_test.time.append(mouse_test.mouseClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=test_screen,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                test_screen.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in test_screen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "test_screen" ---
        for thisComponent in test_screen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for test_screen
        test_screen.tStop = globalClock.getTime(format='float')
        test_screen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('test_screen.stopped', test_screen.tStop)
        # Run 'End Routine' code from test_screen_code
        correct_taps = correct_taps + this_tap
        # store data for test_loop (TrialHandler)
        test_loop.addData('mouse_test.x', mouse_test.x)
        test_loop.addData('mouse_test.y', mouse_test.y)
        test_loop.addData('mouse_test.leftButton', mouse_test.leftButton)
        test_loop.addData('mouse_test.midButton', mouse_test.midButton)
        test_loop.addData('mouse_test.rightButton', mouse_test.rightButton)
        test_loop.addData('mouse_test.time', mouse_test.time)
        test_loop.addData('mouse_test.corr', mouse_test.corr)
        test_loop.addData('mouse_test.clicked_name', mouse_test.clicked_name)
        # the Routine "test_screen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTest_loop as finished
        if hasattr(thisTest_loop, 'status'):
            thisTest_loop.status = FINISHED
        # if awaiting a pause, pause now
        if test_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            test_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed practice_taps repeats of 'test_loop'
    test_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "fb_test" ---
    # create an object to store info about Routine fb_test
    fb_test = data.Routine(
        name='fb_test',
        components=[test_screen_fb_text, fb_test_adv_text, fb_test_adv_resp],
    )
    fb_test.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    test_screen_fb_text.reset()
    test_screen_fb_text.setText(correct_taps_msg + '\n\n' + str(correct_taps) + ' ' + out_of_msg + ' ' + str(practice_taps))
    fb_test_adv_text.setText(adv_msg)
    # create starting attributes for fb_test_adv_resp
    fb_test_adv_resp.keys = []
    fb_test_adv_resp.rt = []
    _fb_test_adv_resp_allKeys = []
    # store start times for fb_test
    fb_test.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    fb_test.tStart = globalClock.getTime(format='float')
    fb_test.status = STARTED
    thisExp.addData('fb_test.started', fb_test.tStart)
    fb_test.maxDuration = None
    # keep track of which components have finished
    fb_testComponents = fb_test.components
    for thisComponent in fb_test.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "fb_test" ---
    fb_test.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *test_screen_fb_text* updates
        
        # if test_screen_fb_text is starting this frame...
        if test_screen_fb_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            test_screen_fb_text.frameNStart = frameN  # exact frame index
            test_screen_fb_text.tStart = t  # local t and not account for scr refresh
            test_screen_fb_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(test_screen_fb_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'test_screen_fb_text.started')
            # update status
            test_screen_fb_text.status = STARTED
            test_screen_fb_text.setAutoDraw(True)
        
        # if test_screen_fb_text is active this frame...
        if test_screen_fb_text.status == STARTED:
            # update params
            pass
        
        # *fb_test_adv_text* updates
        
        # if fb_test_adv_text is starting this frame...
        if fb_test_adv_text.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
            # keep track of start time/frame for later
            fb_test_adv_text.frameNStart = frameN  # exact frame index
            fb_test_adv_text.tStart = t  # local t and not account for scr refresh
            fb_test_adv_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fb_test_adv_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            fb_test_adv_text.status = STARTED
            fb_test_adv_text.setAutoDraw(True)
        
        # if fb_test_adv_text is active this frame...
        if fb_test_adv_text.status == STARTED:
            # update params
            pass
        
        # *fb_test_adv_resp* updates
        waitOnFlip = False
        
        # if fb_test_adv_resp is starting this frame...
        if fb_test_adv_resp.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
            # keep track of start time/frame for later
            fb_test_adv_resp.frameNStart = frameN  # exact frame index
            fb_test_adv_resp.tStart = t  # local t and not account for scr refresh
            fb_test_adv_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fb_test_adv_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            fb_test_adv_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(fb_test_adv_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(fb_test_adv_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if fb_test_adv_resp.status == STARTED and not waitOnFlip:
            theseKeys = fb_test_adv_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _fb_test_adv_resp_allKeys.extend(theseKeys)
            if len(_fb_test_adv_resp_allKeys):
                fb_test_adv_resp.keys = _fb_test_adv_resp_allKeys[-1].name  # just the last key pressed
                fb_test_adv_resp.rt = _fb_test_adv_resp_allKeys[-1].rt
                fb_test_adv_resp.duration = _fb_test_adv_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=fb_test,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            fb_test.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in fb_test.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "fb_test" ---
    for thisComponent in fb_test.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for fb_test
    fb_test.tStop = globalClock.getTime(format='float')
    fb_test.tStopRefresh = tThisFlipGlobal
    thisExp.addData('fb_test.stopped', fb_test.tStop)
    thisExp.nextEntry()
    # the Routine "fb_test" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks_loop = data.TrialHandler2(
        name='blocks_loop',
        nReps=n_blocks, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blocks_loop)  # add the loop to the experiment
    thisBlocks_loop = blocks_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlocks_loop.rgb)
    if thisBlocks_loop != None:
        for paramName in thisBlocks_loop:
            globals()[paramName] = thisBlocks_loop[paramName]
    
    for thisBlocks_loop in blocks_loop:
        blocks_loop.status = STARTED
        if hasattr(thisBlocks_loop, 'status'):
            thisBlocks_loop.status = STARTED
        currentLoop = blocks_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisBlocks_loop.rgb)
        if thisBlocks_loop != None:
            for paramName in thisBlocks_loop:
                globals()[paramName] = thisBlocks_loop[paramName]
        
        # --- Prepare to start Routine "start_block" ---
        # create an object to store info about Routine start_block
        start_block = data.Routine(
            name='start_block',
            components=[block_n_text, start_block_text, start_block_adv_text, start_block_adv_key],
        )
        start_block.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from start_block_code
        current_block += 1
        trial_n = 0
        video_start = 1000
        
        if current_block in [1,3]:
            feedback = True
            reps_per_block = practice_reps
            show_inst = 999
            block_msg = practice_block_msg
        else:
            feedback = False
            reps_per_block = main_blocks_reps
            show_inst = 0
            block_msg = task_block_msg
        
        trials_block = n_IDs * reps_per_block
        
        if current_block < 3:
            if order == '0':
                inst_file = 'instructions_execution.xlsx'
                video_file = 'images/execution_animation.mp4'
            elif order == '1':
                inst_file = 'instructions_imagery.xlsx'
                video_file = 'images/imagery_animation.mp4'
        elif current_block > 2:
            if order == '0':
                inst_file = 'instructions_imagery.xlsx'
                video_file = 'images/imagery_animation.mp4'
            elif order == '1':
                inst_file = 'instructions_execution.xlsx'
                video_file = 'images/execution_animation.mp4'
        
        if inst_file == 'instructions_imagery.xlsx' and current_block not in [1,3]:
            show_back_forth = 9999
        else:
            show_back_forth = 0
        block_n_text.setText(block_n_msg + ' ' + str(current_block) + ' ' + out_of_msg + ' ' + str(n_blocks))
        start_block_text.setText(block_msg)
        start_block_adv_text.setText(adv_msg)
        # create starting attributes for start_block_adv_key
        start_block_adv_key.keys = []
        start_block_adv_key.rt = []
        _start_block_adv_key_allKeys = []
        # store start times for start_block
        start_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        start_block.tStart = globalClock.getTime(format='float')
        start_block.status = STARTED
        thisExp.addData('start_block.started', start_block.tStart)
        start_block.maxDuration = None
        # keep track of which components have finished
        start_blockComponents = start_block.components
        for thisComponent in start_block.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "start_block" ---
        start_block.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlocks_loop, 'status') and thisBlocks_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *block_n_text* updates
            
            # if block_n_text is starting this frame...
            if block_n_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_n_text.frameNStart = frameN  # exact frame index
                block_n_text.tStart = t  # local t and not account for scr refresh
                block_n_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_n_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_n_text.started')
                # update status
                block_n_text.status = STARTED
                block_n_text.setAutoDraw(True)
            
            # if block_n_text is active this frame...
            if block_n_text.status == STARTED:
                # update params
                pass
            
            # *start_block_text* updates
            
            # if start_block_text is starting this frame...
            if start_block_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                start_block_text.frameNStart = frameN  # exact frame index
                start_block_text.tStart = t  # local t and not account for scr refresh
                start_block_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(start_block_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start_block_text.started')
                # update status
                start_block_text.status = STARTED
                start_block_text.setAutoDraw(True)
            
            # if start_block_text is active this frame...
            if start_block_text.status == STARTED:
                # update params
                pass
            
            # *start_block_adv_text* updates
            
            # if start_block_adv_text is starting this frame...
            if start_block_adv_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                start_block_adv_text.frameNStart = frameN  # exact frame index
                start_block_adv_text.tStart = t  # local t and not account for scr refresh
                start_block_adv_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(start_block_adv_text, 'tStartRefresh')  # time at next scr refresh
                # update status
                start_block_adv_text.status = STARTED
                start_block_adv_text.setAutoDraw(True)
            
            # if start_block_adv_text is active this frame...
            if start_block_adv_text.status == STARTED:
                # update params
                pass
            
            # *start_block_adv_key* updates
            waitOnFlip = False
            
            # if start_block_adv_key is starting this frame...
            if start_block_adv_key.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
                # keep track of start time/frame for later
                start_block_adv_key.frameNStart = frameN  # exact frame index
                start_block_adv_key.tStart = t  # local t and not account for scr refresh
                start_block_adv_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(start_block_adv_key, 'tStartRefresh')  # time at next scr refresh
                # update status
                start_block_adv_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(start_block_adv_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(start_block_adv_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if start_block_adv_key.status == STARTED and not waitOnFlip:
                theseKeys = start_block_adv_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _start_block_adv_key_allKeys.extend(theseKeys)
                if len(_start_block_adv_key_allKeys):
                    start_block_adv_key.keys = _start_block_adv_key_allKeys[-1].name  # just the last key pressed
                    start_block_adv_key.rt = _start_block_adv_key_allKeys[-1].rt
                    start_block_adv_key.duration = _start_block_adv_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=start_block,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                start_block.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in start_block.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "start_block" ---
        for thisComponent in start_block.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for start_block
        start_block.tStop = globalClock.getTime(format='float')
        start_block.tStopRefresh = tThisFlipGlobal
        thisExp.addData('start_block.stopped', start_block.tStop)
        # the Routine "start_block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        comp_check_loop = data.TrialHandler2(
            name='comp_check_loop',
            nReps=show_inst, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(comp_check_loop)  # add the loop to the experiment
        thisComp_check_loop = comp_check_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisComp_check_loop.rgb)
        if thisComp_check_loop != None:
            for paramName in thisComp_check_loop:
                globals()[paramName] = thisComp_check_loop[paramName]
        
        for thisComp_check_loop in comp_check_loop:
            comp_check_loop.status = STARTED
            if hasattr(thisComp_check_loop, 'status'):
                thisComp_check_loop.status = STARTED
            currentLoop = comp_check_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisComp_check_loop.rgb)
            if thisComp_check_loop != None:
                for paramName in thisComp_check_loop:
                    globals()[paramName] = thisComp_check_loop[paramName]
            
            # set up handler to look after randomisation of conditions etc
            instructions_loop = data.TrialHandler2(
                name='instructions_loop',
                nReps=1.0, 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions(inst_file), 
                seed=None, 
            )
            thisExp.addLoop(instructions_loop)  # add the loop to the experiment
            thisInstructions_loop = instructions_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisInstructions_loop.rgb)
            if thisInstructions_loop != None:
                for paramName in thisInstructions_loop:
                    globals()[paramName] = thisInstructions_loop[paramName]
            
            for thisInstructions_loop in instructions_loop:
                instructions_loop.status = STARTED
                if hasattr(thisInstructions_loop, 'status'):
                    thisInstructions_loop.status = STARTED
                currentLoop = instructions_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisInstructions_loop.rgb)
                if thisInstructions_loop != None:
                    for paramName in thisInstructions_loop:
                        globals()[paramName] = thisInstructions_loop[paramName]
                
                # --- Prepare to start Routine "instructions" ---
                # create an object to store info about Routine instructions
                instructions = data.Routine(
                    name='instructions',
                    components=[inst_text, inst_adv_resp, inst_image, inst_adv_text, animation_video],
                )
                instructions.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from inst_code
                win.mouseVisible = True
                try:
                    inst_msg = eval(f"inst_msg_{lang_code}")
                except NameError:
                    inst_msg = inst_msg_EN
                
                import numpy as np
                if video == "yes":
                    video_start = 2
                    adv_time = 14
                    if expInfo['handedness'] == "Left":
                        animation_video.size = (-animation_video.size[0], animation_video.size[1])
                else:
                    adv_time = 0.5
                    video_start = 1000
                inst_text.setPos((text_x, text_y))
                inst_text.setText(inst_msg)
                # create starting attributes for inst_adv_resp
                inst_adv_resp.keys = []
                inst_adv_resp.rt = []
                _inst_adv_resp_allKeys = []
                inst_image.setPos((image_x, image_y))
                inst_image.setSize((image_w, image_h))
                inst_image.setImage(inst_pics)
                inst_adv_text.setText(adv_msg)
                animation_video.setMovie(video_file)
                # store start times for instructions
                instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                instructions.tStart = globalClock.getTime(format='float')
                instructions.status = STARTED
                instructions.maxDuration = None
                # keep track of which components have finished
                instructionsComponents = instructions.components
                for thisComponent in instructions.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "instructions" ---
                instructions.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisInstructions_loop, 'status') and thisInstructions_loop.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *inst_text* updates
                    
                    # if inst_text is starting this frame...
                    if inst_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        inst_text.frameNStart = frameN  # exact frame index
                        inst_text.tStart = t  # local t and not account for scr refresh
                        inst_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(inst_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        inst_text.status = STARTED
                        inst_text.setAutoDraw(True)
                    
                    # if inst_text is active this frame...
                    if inst_text.status == STARTED:
                        # update params
                        pass
                    
                    # *inst_adv_resp* updates
                    waitOnFlip = False
                    
                    # if inst_adv_resp is starting this frame...
                    if inst_adv_resp.status == NOT_STARTED and tThisFlip >= adv_time-frameTolerance:
                        # keep track of start time/frame for later
                        inst_adv_resp.frameNStart = frameN  # exact frame index
                        inst_adv_resp.tStart = t  # local t and not account for scr refresh
                        inst_adv_resp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(inst_adv_resp, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        inst_adv_resp.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(inst_adv_resp.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(inst_adv_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if inst_adv_resp.status == STARTED and not waitOnFlip:
                        theseKeys = inst_adv_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                        _inst_adv_resp_allKeys.extend(theseKeys)
                        if len(_inst_adv_resp_allKeys):
                            inst_adv_resp.keys = _inst_adv_resp_allKeys[-1].name  # just the last key pressed
                            inst_adv_resp.rt = _inst_adv_resp_allKeys[-1].rt
                            inst_adv_resp.duration = _inst_adv_resp_allKeys[-1].duration
                            # a response ends the routine
                            continueRoutine = False
                    
                    # *inst_image* updates
                    
                    # if inst_image is starting this frame...
                    if inst_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        inst_image.frameNStart = frameN  # exact frame index
                        inst_image.tStart = t  # local t and not account for scr refresh
                        inst_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(inst_image, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        inst_image.status = STARTED
                        inst_image.setAutoDraw(True)
                    
                    # if inst_image is active this frame...
                    if inst_image.status == STARTED:
                        # update params
                        pass
                    
                    # *inst_adv_text* updates
                    
                    # if inst_adv_text is starting this frame...
                    if inst_adv_text.status == NOT_STARTED and tThisFlip >= adv_time-frameTolerance:
                        # keep track of start time/frame for later
                        inst_adv_text.frameNStart = frameN  # exact frame index
                        inst_adv_text.tStart = t  # local t and not account for scr refresh
                        inst_adv_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(inst_adv_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        inst_adv_text.status = STARTED
                        inst_adv_text.setAutoDraw(True)
                    
                    # if inst_adv_text is active this frame...
                    if inst_adv_text.status == STARTED:
                        # update params
                        pass
                    
                    # *animation_video* updates
                    
                    # if animation_video is starting this frame...
                    if animation_video.status == NOT_STARTED and tThisFlip >= video_start-frameTolerance:
                        # keep track of start time/frame for later
                        animation_video.frameNStart = frameN  # exact frame index
                        animation_video.tStart = t  # local t and not account for scr refresh
                        animation_video.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(animation_video, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'animation_video.started')
                        # update status
                        animation_video.status = STARTED
                        animation_video.setAutoDraw(True)
                        animation_video.play()
                    
                    # if animation_video is stopping this frame...
                    if animation_video.status == STARTED:
                        if bool(False) or animation_video.isFinished:
                            # keep track of stop time/frame for later
                            animation_video.tStop = t  # not accounting for scr refresh
                            animation_video.tStopRefresh = tThisFlipGlobal  # on global time
                            animation_video.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'animation_video.stopped')
                            # update status
                            animation_video.status = FINISHED
                            animation_video.setAutoDraw(False)
                            animation_video.stop()
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=instructions,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        instructions.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in instructions.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "instructions" ---
                for thisComponent in instructions.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for instructions
                instructions.tStop = globalClock.getTime(format='float')
                instructions.tStopRefresh = tThisFlipGlobal
                animation_video.stop()  # ensure movie has stopped at end of Routine
                # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisInstructions_loop as finished
                if hasattr(thisInstructions_loop, 'status'):
                    thisInstructions_loop.status = FINISHED
                # if awaiting a pause, pause now
                if instructions_loop.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    instructions_loop.status = STARTED
            # completed 1.0 repeats of 'instructions_loop'
            instructions_loop.status = FINISHED
            
            
            # --- Prepare to start Routine "comp_check" ---
            # create an object to store info about Routine comp_check
            comp_check = data.Routine(
                name='comp_check',
                components=[comp_check_text, buttonA, textboxA, buttonB, textboxB, buttonC, textboxC, buttonD, textboxD, buttonE, textboxE, buttonF, textboxF, submit_button],
            )
            comp_check.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from comp_check_code
            win.mouseVisible = True
            buttonA.wasClicked = False
            buttonB.wasClicked = False
            buttonC.wasClicked = False
            buttonD.wasClicked = False
            buttonE.wasClicked = False
            buttonF.wasClicked = False
            tickA = tickB = tickC = tickD = tickE = tickF = False
            tick_colorA = tick_colorB = tick_colorC = "black"
            tick_colorD = tick_colorE = tick_colorF = "black"
            y_locs = [0.25, 0.15, 0.05, -0.05, -0.15, -0.25]
            if inst_file == "instructions_execution.xlsx":
                optionA_msg = optionA_exe_msg
                optionB_msg = optionB_exe_msg
                optionC_msg = optionC_exe_msg
                optionD_msg = optionD_exe_msg
                optionE_msg = optionE_exe_msg
                optionF_msg = optionF_exe_msg
            else:
                optionA_msg = optionA_ima_msg
                optionB_msg = optionB_ima_msg
                optionC_msg = optionC_ima_msg
                optionD_msg = optionD_ima_msg
                optionE_msg = optionE_ima_msg
                optionF_msg = optionF_ima_msg
            
            comp_check_text.setText(comp_check_msg)
            buttonA.setPos((-0.5, y_locs[0]))
            # reset buttonA to account for continued clicks & clear times on/off
            buttonA.reset()
            textboxA.reset()
            textboxA.setPos((-0.45, y_locs[0]))
            textboxA.setText(optionA_msg)
            buttonB.setPos((-0.5, y_locs[1]))
            # reset buttonB to account for continued clicks & clear times on/off
            buttonB.reset()
            textboxB.reset()
            textboxB.setPos((-0.45, y_locs[1]))
            textboxB.setText(optionB_msg)
            buttonC.setPos((-0.5, y_locs[2]))
            # reset buttonC to account for continued clicks & clear times on/off
            buttonC.reset()
            textboxC.reset()
            textboxC.setPos((-0.45, y_locs[2]))
            textboxC.setText(optionC_msg)
            buttonD.setPos((-0.5, y_locs[3]))
            # reset buttonD to account for continued clicks & clear times on/off
            buttonD.reset()
            textboxD.reset()
            textboxD.setPos((-0.45, y_locs[3]))
            textboxD.setText(optionD_msg)
            buttonE.setPos((-0.5, y_locs[4]))
            # reset buttonE to account for continued clicks & clear times on/off
            buttonE.reset()
            textboxE.reset()
            textboxE.setPos((-0.45, y_locs[4]))
            textboxE.setText(optionE_msg)
            buttonF.setPos((-0.5, y_locs[5]))
            # reset buttonF to account for continued clicks & clear times on/off
            buttonF.reset()
            textboxF.reset()
            textboxF.setPos((-0.45, y_locs[5]))
            textboxF.setText(optionF_msg)
            submit_button.setText(submit_msg)
            # reset submit_button to account for continued clicks & clear times on/off
            submit_button.reset()
            # store start times for comp_check
            comp_check.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            comp_check.tStart = globalClock.getTime(format='float')
            comp_check.status = STARTED
            thisExp.addData('comp_check.started', comp_check.tStart)
            comp_check.maxDuration = None
            # keep track of which components have finished
            comp_checkComponents = comp_check.components
            for thisComponent in comp_check.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "comp_check" ---
            comp_check.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisComp_check_loop, 'status') and thisComp_check_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from comp_check_code
                if buttonA.isClicked:
                    if not buttonA.wasClicked:
                        tickA = not tickA
                        tick_colorA = "white" if tickA else "black"
                    buttonA.wasClicked = True
                else:
                    buttonA.wasClicked = False
                
                if buttonB.isClicked:
                    if not buttonB.wasClicked:
                        tickB = not tickB
                        tick_colorB = "white" if tickB else "black"
                    buttonB.wasClicked = True
                else:
                    buttonB.wasClicked = False
                
                if buttonC.isClicked:
                    if not buttonC.wasClicked:
                        tickC = not tickC
                        tick_colorC = "white" if tickC else "black"
                    buttonC.wasClicked = True
                else:
                    buttonC.wasClicked = False
                
                if buttonD.isClicked:
                    if not buttonD.wasClicked:
                        tickD = not tickD
                        tick_colorD = "white" if tickD else "black"
                    buttonD.wasClicked = True
                else:
                    buttonD.wasClicked = False
                
                if buttonE.isClicked:
                    if not buttonE.wasClicked:
                        tickE = not tickE
                        tick_colorE = "white" if tickE else "black"
                    buttonE.wasClicked = True
                else:
                    buttonE.wasClicked = False
                
                if buttonF.isClicked:
                    if not buttonF.wasClicked:
                        tickF = not tickF
                        tick_colorF = "white" if tickF else "black"
                    buttonF.wasClicked = True
                else:
                    buttonF.wasClicked = False
                
                # *comp_check_text* updates
                
                # if comp_check_text is starting this frame...
                if comp_check_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    comp_check_text.frameNStart = frameN  # exact frame index
                    comp_check_text.tStart = t  # local t and not account for scr refresh
                    comp_check_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(comp_check_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'comp_check_text.started')
                    # update status
                    comp_check_text.status = STARTED
                    comp_check_text.setAutoDraw(True)
                
                # if comp_check_text is active this frame...
                if comp_check_text.status == STARTED:
                    # update params
                    pass
                # *buttonA* updates
                
                # if buttonA is starting this frame...
                if buttonA.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    buttonA.frameNStart = frameN  # exact frame index
                    buttonA.tStart = t  # local t and not account for scr refresh
                    buttonA.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(buttonA, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonA.started')
                    # update status
                    buttonA.status = STARTED
                    win.callOnFlip(buttonA.buttonClock.reset)
                    buttonA.setAutoDraw(True)
                
                # if buttonA is active this frame...
                if buttonA.status == STARTED:
                    # update params
                    buttonA.setFillColor(tick_colorA, log=False)
                    # check whether buttonA has been pressed
                    if buttonA.isClicked:
                        if not buttonA.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            buttonA.timesOn.append(buttonA.buttonClock.getTime())
                            buttonA.timesOff.append(buttonA.buttonClock.getTime())
                        elif len(buttonA.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            buttonA.timesOff[-1] = buttonA.buttonClock.getTime()
                        if not buttonA.wasClicked:
                            # run callback code when buttonA is clicked
                            tickA = True
                            tick_colorA = "white"
                # take note of whether buttonA was clicked, so that next frame we know if clicks are new
                buttonA.wasClicked = buttonA.isClicked and buttonA.status == STARTED
                
                # *textboxA* updates
                
                # if textboxA is starting this frame...
                if textboxA.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textboxA.frameNStart = frameN  # exact frame index
                    textboxA.tStart = t  # local t and not account for scr refresh
                    textboxA.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textboxA, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textboxA.started')
                    # update status
                    textboxA.status = STARTED
                    textboxA.setAutoDraw(True)
                
                # if textboxA is active this frame...
                if textboxA.status == STARTED:
                    # update params
                    pass
                # *buttonB* updates
                
                # if buttonB is starting this frame...
                if buttonB.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    buttonB.frameNStart = frameN  # exact frame index
                    buttonB.tStart = t  # local t and not account for scr refresh
                    buttonB.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(buttonB, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonB.started')
                    # update status
                    buttonB.status = STARTED
                    win.callOnFlip(buttonB.buttonClock.reset)
                    buttonB.setAutoDraw(True)
                
                # if buttonB is active this frame...
                if buttonB.status == STARTED:
                    # update params
                    buttonB.setFillColor(tick_colorB, log=False)
                    # check whether buttonB has been pressed
                    if buttonB.isClicked:
                        if not buttonB.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            buttonB.timesOn.append(buttonB.buttonClock.getTime())
                            buttonB.timesOff.append(buttonB.buttonClock.getTime())
                        elif len(buttonB.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            buttonB.timesOff[-1] = buttonB.buttonClock.getTime()
                        if not buttonB.wasClicked:
                            # run callback code when buttonB is clicked
                            tickB = True
                            tick_colorB = "white"
                # take note of whether buttonB was clicked, so that next frame we know if clicks are new
                buttonB.wasClicked = buttonB.isClicked and buttonB.status == STARTED
                
                # *textboxB* updates
                
                # if textboxB is starting this frame...
                if textboxB.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textboxB.frameNStart = frameN  # exact frame index
                    textboxB.tStart = t  # local t and not account for scr refresh
                    textboxB.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textboxB, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textboxB.started')
                    # update status
                    textboxB.status = STARTED
                    textboxB.setAutoDraw(True)
                
                # if textboxB is active this frame...
                if textboxB.status == STARTED:
                    # update params
                    pass
                # *buttonC* updates
                
                # if buttonC is starting this frame...
                if buttonC.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    buttonC.frameNStart = frameN  # exact frame index
                    buttonC.tStart = t  # local t and not account for scr refresh
                    buttonC.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(buttonC, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonC.started')
                    # update status
                    buttonC.status = STARTED
                    win.callOnFlip(buttonC.buttonClock.reset)
                    buttonC.setAutoDraw(True)
                
                # if buttonC is active this frame...
                if buttonC.status == STARTED:
                    # update params
                    buttonC.setFillColor(tick_colorC, log=False)
                    # check whether buttonC has been pressed
                    if buttonC.isClicked:
                        if not buttonC.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            buttonC.timesOn.append(buttonC.buttonClock.getTime())
                            buttonC.timesOff.append(buttonC.buttonClock.getTime())
                        elif len(buttonC.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            buttonC.timesOff[-1] = buttonC.buttonClock.getTime()
                        if not buttonC.wasClicked:
                            # run callback code when buttonC is clicked
                            tickC = True
                            tick_colorC = "white"
                # take note of whether buttonC was clicked, so that next frame we know if clicks are new
                buttonC.wasClicked = buttonC.isClicked and buttonC.status == STARTED
                
                # *textboxC* updates
                
                # if textboxC is starting this frame...
                if textboxC.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textboxC.frameNStart = frameN  # exact frame index
                    textboxC.tStart = t  # local t and not account for scr refresh
                    textboxC.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textboxC, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textboxC.started')
                    # update status
                    textboxC.status = STARTED
                    textboxC.setAutoDraw(True)
                
                # if textboxC is active this frame...
                if textboxC.status == STARTED:
                    # update params
                    pass
                # *buttonD* updates
                
                # if buttonD is starting this frame...
                if buttonD.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    buttonD.frameNStart = frameN  # exact frame index
                    buttonD.tStart = t  # local t and not account for scr refresh
                    buttonD.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(buttonD, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonD.started')
                    # update status
                    buttonD.status = STARTED
                    win.callOnFlip(buttonD.buttonClock.reset)
                    buttonD.setAutoDraw(True)
                
                # if buttonD is active this frame...
                if buttonD.status == STARTED:
                    # update params
                    buttonD.setFillColor(tick_colorD, log=False)
                    # check whether buttonD has been pressed
                    if buttonD.isClicked:
                        if not buttonD.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            buttonD.timesOn.append(buttonD.buttonClock.getTime())
                            buttonD.timesOff.append(buttonD.buttonClock.getTime())
                        elif len(buttonD.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            buttonD.timesOff[-1] = buttonD.buttonClock.getTime()
                        if not buttonD.wasClicked:
                            # run callback code when buttonD is clicked
                            tickD = True
                            tick_colorD = "white"
                # take note of whether buttonD was clicked, so that next frame we know if clicks are new
                buttonD.wasClicked = buttonD.isClicked and buttonD.status == STARTED
                
                # *textboxD* updates
                
                # if textboxD is starting this frame...
                if textboxD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textboxD.frameNStart = frameN  # exact frame index
                    textboxD.tStart = t  # local t and not account for scr refresh
                    textboxD.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textboxD, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textboxD.started')
                    # update status
                    textboxD.status = STARTED
                    textboxD.setAutoDraw(True)
                
                # if textboxD is active this frame...
                if textboxD.status == STARTED:
                    # update params
                    pass
                # *buttonE* updates
                
                # if buttonE is starting this frame...
                if buttonE.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    buttonE.frameNStart = frameN  # exact frame index
                    buttonE.tStart = t  # local t and not account for scr refresh
                    buttonE.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(buttonE, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonE.started')
                    # update status
                    buttonE.status = STARTED
                    win.callOnFlip(buttonE.buttonClock.reset)
                    buttonE.setAutoDraw(True)
                
                # if buttonE is active this frame...
                if buttonE.status == STARTED:
                    # update params
                    buttonE.setFillColor(tick_colorE, log=False)
                    # check whether buttonE has been pressed
                    if buttonE.isClicked:
                        if not buttonE.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            buttonE.timesOn.append(buttonE.buttonClock.getTime())
                            buttonE.timesOff.append(buttonE.buttonClock.getTime())
                        elif len(buttonE.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            buttonE.timesOff[-1] = buttonE.buttonClock.getTime()
                        if not buttonE.wasClicked:
                            # run callback code when buttonE is clicked
                            tickE = True
                            tick_colorE = "white"
                # take note of whether buttonE was clicked, so that next frame we know if clicks are new
                buttonE.wasClicked = buttonE.isClicked and buttonE.status == STARTED
                
                # *textboxE* updates
                
                # if textboxE is starting this frame...
                if textboxE.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textboxE.frameNStart = frameN  # exact frame index
                    textboxE.tStart = t  # local t and not account for scr refresh
                    textboxE.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textboxE, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textboxE.started')
                    # update status
                    textboxE.status = STARTED
                    textboxE.setAutoDraw(True)
                
                # if textboxE is active this frame...
                if textboxE.status == STARTED:
                    # update params
                    pass
                # *buttonF* updates
                
                # if buttonF is starting this frame...
                if buttonF.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    buttonF.frameNStart = frameN  # exact frame index
                    buttonF.tStart = t  # local t and not account for scr refresh
                    buttonF.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(buttonF, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buttonF.started')
                    # update status
                    buttonF.status = STARTED
                    win.callOnFlip(buttonF.buttonClock.reset)
                    buttonF.setAutoDraw(True)
                
                # if buttonF is active this frame...
                if buttonF.status == STARTED:
                    # update params
                    buttonF.setFillColor(tick_colorF, log=False)
                    # check whether buttonF has been pressed
                    if buttonF.isClicked:
                        if not buttonF.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            buttonF.timesOn.append(buttonF.buttonClock.getTime())
                            buttonF.timesOff.append(buttonF.buttonClock.getTime())
                        elif len(buttonF.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            buttonF.timesOff[-1] = buttonF.buttonClock.getTime()
                        if not buttonF.wasClicked:
                            # run callback code when buttonF is clicked
                            tickF = True
                            tick_colorF = "white"
                # take note of whether buttonF was clicked, so that next frame we know if clicks are new
                buttonF.wasClicked = buttonF.isClicked and buttonF.status == STARTED
                
                # *textboxF* updates
                
                # if textboxF is starting this frame...
                if textboxF.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textboxF.frameNStart = frameN  # exact frame index
                    textboxF.tStart = t  # local t and not account for scr refresh
                    textboxF.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textboxF, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textboxF.started')
                    # update status
                    textboxF.status = STARTED
                    textboxF.setAutoDraw(True)
                
                # if textboxF is active this frame...
                if textboxF.status == STARTED:
                    # update params
                    pass
                # *submit_button* updates
                
                # if submit_button is starting this frame...
                if submit_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    submit_button.frameNStart = frameN  # exact frame index
                    submit_button.tStart = t  # local t and not account for scr refresh
                    submit_button.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(submit_button, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'submit_button.started')
                    # update status
                    submit_button.status = STARTED
                    win.callOnFlip(submit_button.buttonClock.reset)
                    submit_button.setAutoDraw(True)
                
                # if submit_button is active this frame...
                if submit_button.status == STARTED:
                    # update params
                    pass
                    # check whether submit_button has been pressed
                    if submit_button.isClicked:
                        if not submit_button.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            submit_button.timesOn.append(submit_button.buttonClock.getTime())
                            submit_button.timesOff.append(submit_button.buttonClock.getTime())
                        elif len(submit_button.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            submit_button.timesOff[-1] = submit_button.buttonClock.getTime()
                        if not submit_button.wasClicked:
                            # end routine when submit_button is clicked
                            continueRoutine = False
                        if not submit_button.wasClicked:
                            # run callback code when submit_button is clicked
                            pass
                # take note of whether submit_button was clicked, so that next frame we know if clicks are new
                submit_button.wasClicked = submit_button.isClicked and submit_button.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=comp_check,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    comp_check.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in comp_check.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "comp_check" ---
            for thisComponent in comp_check.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for comp_check
            comp_check.tStop = globalClock.getTime(format='float')
            comp_check.tStopRefresh = tThisFlipGlobal
            thisExp.addData('comp_check.stopped', comp_check.tStop)
            # Run 'End Routine' code from comp_check_code
            if inst_file == "instructions_execution.xlsx":
                if tickC and tickF and not (tickA or tickB or tickD or tickE):
                    response_correct = True
                else:
                    response_correct = False
            else:
                if tickA and tickB and tickE and tickF and not (tickC or tickD):
                    response_correct = True
                else:
                    response_correct = False
            comp_check_loop.addData('buttonA.numClicks', buttonA.numClicks)
            if buttonA.numClicks:
               comp_check_loop.addData('buttonA.timesOn', buttonA.timesOn)
               comp_check_loop.addData('buttonA.timesOff', buttonA.timesOff)
            else:
               comp_check_loop.addData('buttonA.timesOn', "")
               comp_check_loop.addData('buttonA.timesOff', "")
            comp_check_loop.addData('buttonB.numClicks', buttonB.numClicks)
            if buttonB.numClicks:
               comp_check_loop.addData('buttonB.timesOn', buttonB.timesOn)
               comp_check_loop.addData('buttonB.timesOff', buttonB.timesOff)
            else:
               comp_check_loop.addData('buttonB.timesOn', "")
               comp_check_loop.addData('buttonB.timesOff', "")
            comp_check_loop.addData('buttonC.numClicks', buttonC.numClicks)
            if buttonC.numClicks:
               comp_check_loop.addData('buttonC.timesOn', buttonC.timesOn)
               comp_check_loop.addData('buttonC.timesOff', buttonC.timesOff)
            else:
               comp_check_loop.addData('buttonC.timesOn', "")
               comp_check_loop.addData('buttonC.timesOff', "")
            comp_check_loop.addData('buttonD.numClicks', buttonD.numClicks)
            if buttonD.numClicks:
               comp_check_loop.addData('buttonD.timesOn', buttonD.timesOn)
               comp_check_loop.addData('buttonD.timesOff', buttonD.timesOff)
            else:
               comp_check_loop.addData('buttonD.timesOn', "")
               comp_check_loop.addData('buttonD.timesOff', "")
            comp_check_loop.addData('buttonE.numClicks', buttonE.numClicks)
            if buttonE.numClicks:
               comp_check_loop.addData('buttonE.timesOn', buttonE.timesOn)
               comp_check_loop.addData('buttonE.timesOff', buttonE.timesOff)
            else:
               comp_check_loop.addData('buttonE.timesOn', "")
               comp_check_loop.addData('buttonE.timesOff', "")
            comp_check_loop.addData('buttonF.numClicks', buttonF.numClicks)
            if buttonF.numClicks:
               comp_check_loop.addData('buttonF.timesOn', buttonF.timesOn)
               comp_check_loop.addData('buttonF.timesOff', buttonF.timesOff)
            else:
               comp_check_loop.addData('buttonF.timesOn', "")
               comp_check_loop.addData('buttonF.timesOff', "")
            comp_check_loop.addData('submit_button.numClicks', submit_button.numClicks)
            if submit_button.numClicks:
               comp_check_loop.addData('submit_button.timesOn', submit_button.timesOn)
               comp_check_loop.addData('submit_button.timesOff', submit_button.timesOff)
            else:
               comp_check_loop.addData('submit_button.timesOn', "")
               comp_check_loop.addData('submit_button.timesOff', "")
            # the Routine "comp_check" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "check_fb" ---
            # create an object to store info about Routine check_fb
            check_fb = data.Routine(
                name='check_fb',
                components=[corr_incorr_fb, check_fb_text, check_fb_adv_text, check_fb_adv_resp],
            )
            check_fb.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from fb_check_code
            if response_correct == True:
                check_fb_msg = check_correct_msg
                color_text = "green"
                corr_incorr_msg = correct_msg
            else:
                check_fb_msg = check_incorrect_msg
                color_text = "red"
                corr_incorr_msg = incorrect_msg
            corr_incorr_fb.setColor(color_text, colorSpace='rgb')
            corr_incorr_fb.setText(corr_incorr_msg)
            check_fb_text.setText(check_fb_msg)
            check_fb_adv_text.setText(adv_msg)
            # create starting attributes for check_fb_adv_resp
            check_fb_adv_resp.keys = []
            check_fb_adv_resp.rt = []
            _check_fb_adv_resp_allKeys = []
            # store start times for check_fb
            check_fb.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            check_fb.tStart = globalClock.getTime(format='float')
            check_fb.status = STARTED
            thisExp.addData('check_fb.started', check_fb.tStart)
            check_fb.maxDuration = None
            # keep track of which components have finished
            check_fbComponents = check_fb.components
            for thisComponent in check_fb.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "check_fb" ---
            check_fb.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisComp_check_loop, 'status') and thisComp_check_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *corr_incorr_fb* updates
                
                # if corr_incorr_fb is starting this frame...
                if corr_incorr_fb.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    corr_incorr_fb.frameNStart = frameN  # exact frame index
                    corr_incorr_fb.tStart = t  # local t and not account for scr refresh
                    corr_incorr_fb.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(corr_incorr_fb, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'corr_incorr_fb.started')
                    # update status
                    corr_incorr_fb.status = STARTED
                    corr_incorr_fb.setAutoDraw(True)
                
                # if corr_incorr_fb is active this frame...
                if corr_incorr_fb.status == STARTED:
                    # update params
                    pass
                
                # *check_fb_text* updates
                
                # if check_fb_text is starting this frame...
                if check_fb_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    check_fb_text.frameNStart = frameN  # exact frame index
                    check_fb_text.tStart = t  # local t and not account for scr refresh
                    check_fb_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(check_fb_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'check_fb_text.started')
                    # update status
                    check_fb_text.status = STARTED
                    check_fb_text.setAutoDraw(True)
                
                # if check_fb_text is active this frame...
                if check_fb_text.status == STARTED:
                    # update params
                    pass
                
                # *check_fb_adv_text* updates
                
                # if check_fb_adv_text is starting this frame...
                if check_fb_adv_text.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                    # keep track of start time/frame for later
                    check_fb_adv_text.frameNStart = frameN  # exact frame index
                    check_fb_adv_text.tStart = t  # local t and not account for scr refresh
                    check_fb_adv_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(check_fb_adv_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    check_fb_adv_text.status = STARTED
                    check_fb_adv_text.setAutoDraw(True)
                
                # if check_fb_adv_text is active this frame...
                if check_fb_adv_text.status == STARTED:
                    # update params
                    pass
                
                # *check_fb_adv_resp* updates
                waitOnFlip = False
                
                # if check_fb_adv_resp is starting this frame...
                if check_fb_adv_resp.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                    # keep track of start time/frame for later
                    check_fb_adv_resp.frameNStart = frameN  # exact frame index
                    check_fb_adv_resp.tStart = t  # local t and not account for scr refresh
                    check_fb_adv_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(check_fb_adv_resp, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    check_fb_adv_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(check_fb_adv_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(check_fb_adv_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if check_fb_adv_resp.status == STARTED and not waitOnFlip:
                    theseKeys = check_fb_adv_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _check_fb_adv_resp_allKeys.extend(theseKeys)
                    if len(_check_fb_adv_resp_allKeys):
                        check_fb_adv_resp.keys = _check_fb_adv_resp_allKeys[-1].name  # just the last key pressed
                        check_fb_adv_resp.rt = _check_fb_adv_resp_allKeys[-1].rt
                        check_fb_adv_resp.duration = _check_fb_adv_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=check_fb,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    check_fb.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in check_fb.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "check_fb" ---
            for thisComponent in check_fb.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for check_fb
            check_fb.tStop = globalClock.getTime(format='float')
            check_fb.tStopRefresh = tThisFlipGlobal
            thisExp.addData('check_fb.stopped', check_fb.tStop)
            # Run 'End Routine' code from fb_check_code
            
            if response_correct == True:
                comp_check_loop.finished = True
            # the Routine "check_fb" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisComp_check_loop as finished
            if hasattr(thisComp_check_loop, 'status'):
                thisComp_check_loop.status = FINISHED
            # if awaiting a pause, pause now
            if comp_check_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                comp_check_loop.status = STARTED
        # completed show_inst repeats of 'comp_check_loop'
        comp_check_loop.status = FINISHED
        
        
        # --- Prepare to start Routine "reminder" ---
        # create an object to store info about Routine reminder
        reminder = data.Routine(
            name='reminder',
            components=[reminder_title, reminder_text, reminder_adv_text, reminder_adv_resp],
        )
        reminder.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from reminder_code
        if inst_file == "instructions_execution.xlsx":
            reminder_main_msg = reminder_exe_msg
        else:
            reminder_main_msg = reminder_ima_msg
        reminder_title.setText(reminder_msg)
        reminder_text.setText(reminder_main_msg)
        reminder_adv_text.setText(adv_msg)
        # create starting attributes for reminder_adv_resp
        reminder_adv_resp.keys = []
        reminder_adv_resp.rt = []
        _reminder_adv_resp_allKeys = []
        # store start times for reminder
        reminder.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        reminder.tStart = globalClock.getTime(format='float')
        reminder.status = STARTED
        thisExp.addData('reminder.started', reminder.tStart)
        reminder.maxDuration = None
        # keep track of which components have finished
        reminderComponents = reminder.components
        for thisComponent in reminder.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "reminder" ---
        reminder.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlocks_loop, 'status') and thisBlocks_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *reminder_title* updates
            
            # if reminder_title is starting this frame...
            if reminder_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reminder_title.frameNStart = frameN  # exact frame index
                reminder_title.tStart = t  # local t and not account for scr refresh
                reminder_title.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reminder_title, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reminder_title.started')
                # update status
                reminder_title.status = STARTED
                reminder_title.setAutoDraw(True)
            
            # if reminder_title is active this frame...
            if reminder_title.status == STARTED:
                # update params
                pass
            
            # *reminder_text* updates
            
            # if reminder_text is starting this frame...
            if reminder_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reminder_text.frameNStart = frameN  # exact frame index
                reminder_text.tStart = t  # local t and not account for scr refresh
                reminder_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reminder_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reminder_text.started')
                # update status
                reminder_text.status = STARTED
                reminder_text.setAutoDraw(True)
            
            # if reminder_text is active this frame...
            if reminder_text.status == STARTED:
                # update params
                pass
            
            # *reminder_adv_text* updates
            
            # if reminder_adv_text is starting this frame...
            if reminder_adv_text.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                reminder_adv_text.frameNStart = frameN  # exact frame index
                reminder_adv_text.tStart = t  # local t and not account for scr refresh
                reminder_adv_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reminder_adv_text, 'tStartRefresh')  # time at next scr refresh
                # update status
                reminder_adv_text.status = STARTED
                reminder_adv_text.setAutoDraw(True)
            
            # if reminder_adv_text is active this frame...
            if reminder_adv_text.status == STARTED:
                # update params
                pass
            
            # *reminder_adv_resp* updates
            waitOnFlip = False
            
            # if reminder_adv_resp is starting this frame...
            if reminder_adv_resp.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                reminder_adv_resp.frameNStart = frameN  # exact frame index
                reminder_adv_resp.tStart = t  # local t and not account for scr refresh
                reminder_adv_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reminder_adv_resp, 'tStartRefresh')  # time at next scr refresh
                # update status
                reminder_adv_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(reminder_adv_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(reminder_adv_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if reminder_adv_resp.status == STARTED and not waitOnFlip:
                theseKeys = reminder_adv_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _reminder_adv_resp_allKeys.extend(theseKeys)
                if len(_reminder_adv_resp_allKeys):
                    reminder_adv_resp.keys = _reminder_adv_resp_allKeys[-1].name  # just the last key pressed
                    reminder_adv_resp.rt = _reminder_adv_resp_allKeys[-1].rt
                    reminder_adv_resp.duration = _reminder_adv_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=reminder,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                reminder.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reminder.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "reminder" ---
        for thisComponent in reminder.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for reminder
        reminder.tStop = globalClock.getTime(format='float')
        reminder.tStopRefresh = tThisFlipGlobal
        thisExp.addData('reminder.stopped', reminder.tStop)
        # the Routine "reminder" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_loop = data.TrialHandler2(
            name='trials_loop',
            nReps=reps_per_block, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('conditions.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(trials_loop)  # add the loop to the experiment
        thisTrials_loop = trials_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
        if thisTrials_loop != None:
            for paramName in thisTrials_loop:
                globals()[paramName] = thisTrials_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrials_loop in trials_loop:
            trials_loop.status = STARTED
            if hasattr(thisTrials_loop, 'status'):
                thisTrials_loop.status = STARTED
            currentLoop = trials_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
            if thisTrials_loop != None:
                for paramName in thisTrials_loop:
                    globals()[paramName] = thisTrials_loop[paramName]
            
            # --- Prepare to start Routine "setup" ---
            # create an object to store info about Routine setup
            setup = data.Routine(
                name='setup',
                components=[press_space_text, start_flag, flag_image, setup_resp],
            )
            setup.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from setup_code
            # Update target diameters if this is a practice or main task block
            if current_block in [1,3]:
                # moderate difficulties for practice blocks
                target_diameters = [0.03, 0.06, 0.09, 0.12, 0.15]
            else:
                # actual difficulties otherwise
                target_diameters = [0.01176, 0.02353, 0.05, 0.1, 0.20588]
            thisExp.addData("target_diameters", target_diameters)
            # Get the level from the trial conditions
            level_index = int(difficulty_index)  # difficulty_index is column from excel file
            target_diameter = target_diameters[level_index]
            # Calculate center-to-target distance for this target diameter
            center_to_target = (start_diameter / 2) + distance + (target_diameter / 2)
            # Calculate ID
            ID = np.log2((2 * center_to_target) / target_diameter)
            ID = np.repeat(round(ID, 2), 5) # because we have 5 targets and 5 movement times
            # Store positions and sizes
            target_positions = []
            target_sizes = []
            for theta in angles_rad:
                angle_corrected = theta + np.pi/2  # make 0 degrees point up
                x = start_x + center_to_target * np.cos(angle_corrected)
                y = start_y + center_to_target * np.sin(angle_corrected)
                target_positions.append((x, y))
                target_sizes.append((target_diameter, target_diameter))
            # Assign to components
            target_1_pos = target_positions[0]
            target_1_size = target_sizes[0]
            target_2_pos = target_positions[1]
            target_2_size = target_sizes[1]
            target_3_pos = target_positions[2]
            target_3_size = target_sizes[2]
            target_4_pos = target_positions[3]
            target_4_size = target_sizes[3]
            target_5_pos = target_positions[4]
            target_5_size = target_sizes[4]
            
            press_space_text.setText(adv_msg)
            start_flag.setPos((start_x, start_y))
            start_flag.setSize((start_diameter, start_diameter))
            flag_image.setPos((start_x, start_y))
            flag_image.setSize((start_diameter-0.03, start_diameter-0.03))
            # create starting attributes for setup_resp
            setup_resp.keys = []
            setup_resp.rt = []
            _setup_resp_allKeys = []
            # store start times for setup
            setup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            setup.tStart = globalClock.getTime(format='float')
            setup.status = STARTED
            thisExp.addData('setup.started', setup.tStart)
            setup.maxDuration = None
            # keep track of which components have finished
            setupComponents = setup.components
            for thisComponent in setup.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "setup" ---
            setup.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_loop, 'status') and thisTrials_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *press_space_text* updates
                
                # if press_space_text is starting this frame...
                if press_space_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    press_space_text.frameNStart = frameN  # exact frame index
                    press_space_text.tStart = t  # local t and not account for scr refresh
                    press_space_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(press_space_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'press_space_text.started')
                    # update status
                    press_space_text.status = STARTED
                    press_space_text.setAutoDraw(True)
                
                # if press_space_text is active this frame...
                if press_space_text.status == STARTED:
                    # update params
                    pass
                
                # *start_flag* updates
                
                # if start_flag is starting this frame...
                if start_flag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    start_flag.frameNStart = frameN  # exact frame index
                    start_flag.tStart = t  # local t and not account for scr refresh
                    start_flag.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(start_flag, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'start_flag.started')
                    # update status
                    start_flag.status = STARTED
                    start_flag.setAutoDraw(True)
                
                # if start_flag is active this frame...
                if start_flag.status == STARTED:
                    # update params
                    pass
                
                # *flag_image* updates
                
                # if flag_image is starting this frame...
                if flag_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    flag_image.frameNStart = frameN  # exact frame index
                    flag_image.tStart = t  # local t and not account for scr refresh
                    flag_image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(flag_image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'flag_image.started')
                    # update status
                    flag_image.status = STARTED
                    flag_image.setAutoDraw(True)
                
                # if flag_image is active this frame...
                if flag_image.status == STARTED:
                    # update params
                    pass
                
                # *setup_resp* updates
                waitOnFlip = False
                
                # if setup_resp is starting this frame...
                if setup_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    setup_resp.frameNStart = frameN  # exact frame index
                    setup_resp.tStart = t  # local t and not account for scr refresh
                    setup_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(setup_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'setup_resp.started')
                    # update status
                    setup_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(setup_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(setup_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if setup_resp.status == STARTED and not waitOnFlip:
                    theseKeys = setup_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _setup_resp_allKeys.extend(theseKeys)
                    if len(_setup_resp_allKeys):
                        setup_resp.keys = _setup_resp_allKeys[0].name  # just the first key pressed
                        setup_resp.rt = _setup_resp_allKeys[0].rt
                        setup_resp.duration = _setup_resp_allKeys[0].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=setup,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    setup.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in setup.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "setup" ---
            for thisComponent in setup.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for setup
            setup.tStop = globalClock.getTime(format='float')
            setup.tStopRefresh = tThisFlipGlobal
            thisExp.addData('setup.stopped', setup.tStop)
            # check responses
            if setup_resp.keys in ['', [], None]:  # No response was made
                setup_resp.keys = None
            trials_loop.addData('setup_resp.keys',setup_resp.keys)
            if setup_resp.keys != None:  # we had a response
                trials_loop.addData('setup_resp.rt', setup_resp.rt)
                trials_loop.addData('setup_resp.duration', setup_resp.duration)
            # the Routine "setup" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "countdown" ---
            # create an object to store info about Routine countdown
            countdown = data.Routine(
                name='countdown',
                components=[start_count, target_1_count, target_2_count, target_3_count, target_4_count, target_5_count, rect_countdown, countdown_3, countdown_2, countdown_1],
            )
            countdown.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            start_count.setPos((start_x, start_y))
            start_count.setSize((start_diameter, start_diameter))
            target_1_count.setPos(target_5_pos)
            target_1_count.setSize(target_1_size)
            target_2_count.setPos(target_4_pos)
            target_2_count.setSize(target_2_size)
            target_3_count.setPos(target_3_pos)
            target_3_count.setSize(target_3_size)
            target_4_count.setPos(target_2_pos)
            target_4_count.setSize(target_4_size)
            target_5_count.setPos(target_1_pos)
            target_5_count.setSize(target_5_size)
            # store start times for countdown
            countdown.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            countdown.tStart = globalClock.getTime(format='float')
            countdown.status = STARTED
            thisExp.addData('countdown.started', countdown.tStart)
            countdown.maxDuration = 3
            # keep track of which components have finished
            countdownComponents = countdown.components
            for thisComponent in countdown.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "countdown" ---
            countdown.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_loop, 'status') and thisTrials_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > countdown.maxDuration-frameTolerance:
                    countdown.maxDurationReached = True
                    continueRoutine = False
                
                # *start_count* updates
                
                # if start_count is starting this frame...
                if start_count.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    start_count.frameNStart = frameN  # exact frame index
                    start_count.tStart = t  # local t and not account for scr refresh
                    start_count.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(start_count, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'start_count.started')
                    # update status
                    start_count.status = STARTED
                    start_count.setAutoDraw(True)
                
                # if start_count is active this frame...
                if start_count.status == STARTED:
                    # update params
                    pass
                
                # *target_1_count* updates
                
                # if target_1_count is starting this frame...
                if target_1_count.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_1_count.frameNStart = frameN  # exact frame index
                    target_1_count.tStart = t  # local t and not account for scr refresh
                    target_1_count.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_1_count, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_1_count.started')
                    # update status
                    target_1_count.status = STARTED
                    target_1_count.setAutoDraw(True)
                
                # if target_1_count is active this frame...
                if target_1_count.status == STARTED:
                    # update params
                    target_1_count.setFillColor(target_1_color, log=False)
                    target_1_count.setLineColor(target_1_color, log=False)
                
                # *target_2_count* updates
                
                # if target_2_count is starting this frame...
                if target_2_count.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_2_count.frameNStart = frameN  # exact frame index
                    target_2_count.tStart = t  # local t and not account for scr refresh
                    target_2_count.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_2_count, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_2_count.started')
                    # update status
                    target_2_count.status = STARTED
                    target_2_count.setAutoDraw(True)
                
                # if target_2_count is active this frame...
                if target_2_count.status == STARTED:
                    # update params
                    target_2_count.setFillColor(target_2_color, log=False)
                    target_2_count.setLineColor(target_2_color, log=False)
                
                # *target_3_count* updates
                
                # if target_3_count is starting this frame...
                if target_3_count.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_3_count.frameNStart = frameN  # exact frame index
                    target_3_count.tStart = t  # local t and not account for scr refresh
                    target_3_count.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_3_count, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_3_count.started')
                    # update status
                    target_3_count.status = STARTED
                    target_3_count.setAutoDraw(True)
                
                # if target_3_count is active this frame...
                if target_3_count.status == STARTED:
                    # update params
                    target_3_count.setFillColor(target_3_color, log=False)
                    target_3_count.setLineColor(target_3_color, log=False)
                
                # *target_4_count* updates
                
                # if target_4_count is starting this frame...
                if target_4_count.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_4_count.frameNStart = frameN  # exact frame index
                    target_4_count.tStart = t  # local t and not account for scr refresh
                    target_4_count.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_4_count, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_4_count.started')
                    # update status
                    target_4_count.status = STARTED
                    target_4_count.setAutoDraw(True)
                
                # if target_4_count is active this frame...
                if target_4_count.status == STARTED:
                    # update params
                    target_4_count.setFillColor(target_4_color, log=False)
                    target_4_count.setLineColor(target_4_color, log=False)
                
                # *target_5_count* updates
                
                # if target_5_count is starting this frame...
                if target_5_count.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_5_count.frameNStart = frameN  # exact frame index
                    target_5_count.tStart = t  # local t and not account for scr refresh
                    target_5_count.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_5_count, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_5_count.started')
                    # update status
                    target_5_count.status = STARTED
                    target_5_count.setAutoDraw(True)
                
                # if target_5_count is active this frame...
                if target_5_count.status == STARTED:
                    # update params
                    target_5_count.setFillColor(target_5_color, log=False)
                    target_5_count.setLineColor(target_5_color, log=False)
                
                # *rect_countdown* updates
                
                # if rect_countdown is starting this frame...
                if rect_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    rect_countdown.frameNStart = frameN  # exact frame index
                    rect_countdown.tStart = t  # local t and not account for scr refresh
                    rect_countdown.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(rect_countdown, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect_countdown.started')
                    # update status
                    rect_countdown.status = STARTED
                    rect_countdown.setAutoDraw(True)
                
                # if rect_countdown is active this frame...
                if rect_countdown.status == STARTED:
                    # update params
                    pass
                
                # if rect_countdown is stopping this frame...
                if rect_countdown.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > rect_countdown.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        rect_countdown.tStop = t  # not accounting for scr refresh
                        rect_countdown.tStopRefresh = tThisFlipGlobal  # on global time
                        rect_countdown.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rect_countdown.stopped')
                        # update status
                        rect_countdown.status = FINISHED
                        rect_countdown.setAutoDraw(False)
                
                # *countdown_3* updates
                
                # if countdown_3 is starting this frame...
                if countdown_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    countdown_3.frameNStart = frameN  # exact frame index
                    countdown_3.tStart = t  # local t and not account for scr refresh
                    countdown_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(countdown_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'countdown_3.started')
                    # update status
                    countdown_3.status = STARTED
                    countdown_3.setAutoDraw(True)
                
                # if countdown_3 is active this frame...
                if countdown_3.status == STARTED:
                    # update params
                    pass
                
                # if countdown_3 is stopping this frame...
                if countdown_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > countdown_3.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        countdown_3.tStop = t  # not accounting for scr refresh
                        countdown_3.tStopRefresh = tThisFlipGlobal  # on global time
                        countdown_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'countdown_3.stopped')
                        # update status
                        countdown_3.status = FINISHED
                        countdown_3.setAutoDraw(False)
                
                # *countdown_2* updates
                
                # if countdown_2 is starting this frame...
                if countdown_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                    # keep track of start time/frame for later
                    countdown_2.frameNStart = frameN  # exact frame index
                    countdown_2.tStart = t  # local t and not account for scr refresh
                    countdown_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(countdown_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'countdown_2.started')
                    # update status
                    countdown_2.status = STARTED
                    countdown_2.setAutoDraw(True)
                
                # if countdown_2 is active this frame...
                if countdown_2.status == STARTED:
                    # update params
                    pass
                
                # if countdown_2 is stopping this frame...
                if countdown_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > countdown_2.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        countdown_2.tStop = t  # not accounting for scr refresh
                        countdown_2.tStopRefresh = tThisFlipGlobal  # on global time
                        countdown_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'countdown_2.stopped')
                        # update status
                        countdown_2.status = FINISHED
                        countdown_2.setAutoDraw(False)
                
                # *countdown_1* updates
                
                # if countdown_1 is starting this frame...
                if countdown_1.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                    # keep track of start time/frame for later
                    countdown_1.frameNStart = frameN  # exact frame index
                    countdown_1.tStart = t  # local t and not account for scr refresh
                    countdown_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(countdown_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'countdown_1.started')
                    # update status
                    countdown_1.status = STARTED
                    countdown_1.setAutoDraw(True)
                
                # if countdown_1 is active this frame...
                if countdown_1.status == STARTED:
                    # update params
                    pass
                
                # if countdown_1 is stopping this frame...
                if countdown_1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > countdown_1.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        countdown_1.tStop = t  # not accounting for scr refresh
                        countdown_1.tStopRefresh = tThisFlipGlobal  # on global time
                        countdown_1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'countdown_1.stopped')
                        # update status
                        countdown_1.status = FINISHED
                        countdown_1.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=countdown,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    countdown.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in countdown.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "countdown" ---
            for thisComponent in countdown.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for countdown
            countdown.tStop = globalClock.getTime(format='float')
            countdown.tStopRefresh = tThisFlipGlobal
            thisExp.addData('countdown.stopped', countdown.tStop)
            # the Routine "countdown" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[start, target_1, target_2, target_3, target_4, target_5, mouse, cursor, space_resp, key_resp],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from trial_code
            mouse.setVisible(False)
            # set position of mouse to center, use it carefully as it might change the x-y coordinate system
            mouse.setPos([0, 0])
            
            if feedback == True:
                cursor_size = 0.001
            else:
                cursor_size = 0
            
            # Expected sequence of taps
            tap_sequence = [start, target_1, start, target_2, start, target_3, start, target_4, start, target_5, start]
            tap_sequence_names = ['start', 'target_1', 'start', 'target_2', 'start', 'target_3', 'start', 'target_4', 'start', 'target_5', 'start']
            tap_sequence_index = 0
            valid_targets = [start, target_1, target_2, target_3, target_4, target_5]
            
            # Initialize colors
            cursor_color = 'black'
            target_1_color = target_2_color = target_3_color = target_4_color = target_5_color = 'grey'
            
            # Time settings
            color_reset_time = None
            color_flash_time = 0.2 
            
            # Tap settings
            last_pos = mouse.getPos()
            last_time = core.getTime()
            tap_positions = []
            tap_times = []
            tapped_names = []
            routine_clock = core.Clock()
            
            # Space settings
            previous_key_count = 0
            current_key_count = 0
            space_resp.keys = []
            key_count = 0
            
            
            
            start.setPos((start_x, start_y))
            start.setSize((start_diameter, start_diameter))
            target_1.setPos(target_5_pos)
            target_1.setSize(target_1_size)
            target_2.setPos(target_4_pos)
            target_2.setSize(target_2_size)
            target_3.setPos(target_3_pos)
            target_3.setSize(target_3_size)
            target_4.setPos(target_2_pos)
            target_4.setSize(target_4_size)
            target_5.setPos(target_1_pos)
            target_5.setSize(target_5_size)
            # setup some python lists for storing info about the mouse
            mouse.x = []
            mouse.y = []
            mouse.leftButton = []
            mouse.midButton = []
            mouse.rightButton = []
            mouse.time = []
            mouse.corr = []
            mouse.clicked_name = []
            gotValidClick = False  # until a click is received
            cursor.setSize((cursor_size, cursor_size))
            # create starting attributes for space_resp
            space_resp.keys = []
            space_resp.rt = []
            _space_resp_allKeys = []
            # create starting attributes for key_resp
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_loop, 'status') and thisTrials_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from trial_code
                # Get current mouse position
                cursor_x, cursor_y = mouse.getPos()
                new_pos = mouse.getPos()
                
                # Detect a "tap" when the mouse moves and enough time has passed
                if tuple(new_pos) != tuple(last_pos) and core.getTime() - last_time > 0.1:
                    valid_tap = False
                    tapped_name = None  # default: no valid target was tapped
                    # Check if the tap was on a valid target
                    for target in valid_targets:
                        if target.contains(new_pos):
                            valid_tap = True
                            tapped_name = target.name  # store name of the target
                            break
                    # Store tap data
                    tap_positions.append(new_pos)                # tap position
                    tap_times.append(routine_clock.getTime())             # tap time from start of routine
                    tapped_names.append(tapped_name)             # target name (or None)
                    # Change cursor color based on tap validity
                    if valid_tap:
                        cursor_color = 'green'
                    else:
                        cursor_color = 'red'
                    # Set time to reset cursor color
                    color_reset_time = t + color_flash_time
                    last_time = core.getTime()
                
                # Update last position of the mouse
                last_pos = new_pos
                
                # Reset cursor color after flash duration
                if color_reset_time is not None and t >= color_reset_time:
                    cursor_color = 'black'
                    color_reset_time = None
                
                # At the beginning, no key has been pressed
                # We initialize the list of key presses
                if space_resp.keys is None:
                    space_resp.keys = []
                
                # set variable to count the number of keys
                current_key_count = len(space_resp.keys) # use the native variable from component
                
                # Check if a key has been pressed
                if current_key_count > previous_key_count:
                    key_count += 1
                else:
                    key_count = key_count
                
                # Update the previous key count for the next frame
                previous_key_count = current_key_count
                
                # End trial when reaching expected number of keys
                if key_count >= key_presses:
                    continueRoutine = False
                else:
                    continueRoutine = True
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                # *start* updates
                
                # if start is starting this frame...
                if start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    start.frameNStart = frameN  # exact frame index
                    start.tStart = t  # local t and not account for scr refresh
                    start.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(start, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'start.started')
                    # update status
                    start.status = STARTED
                    start.setAutoDraw(True)
                
                # if start is active this frame...
                if start.status == STARTED:
                    # update params
                    pass
                
                # *target_1* updates
                
                # if target_1 is starting this frame...
                if target_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_1.frameNStart = frameN  # exact frame index
                    target_1.tStart = t  # local t and not account for scr refresh
                    target_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_1.started')
                    # update status
                    target_1.status = STARTED
                    target_1.setAutoDraw(True)
                
                # if target_1 is active this frame...
                if target_1.status == STARTED:
                    # update params
                    target_1.setFillColor(target_1_color, log=False)
                    target_1.setLineColor(target_1_color, log=False)
                
                # *target_2* updates
                
                # if target_2 is starting this frame...
                if target_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_2.frameNStart = frameN  # exact frame index
                    target_2.tStart = t  # local t and not account for scr refresh
                    target_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_2.started')
                    # update status
                    target_2.status = STARTED
                    target_2.setAutoDraw(True)
                
                # if target_2 is active this frame...
                if target_2.status == STARTED:
                    # update params
                    target_2.setFillColor(target_2_color, log=False)
                    target_2.setLineColor(target_2_color, log=False)
                
                # *target_3* updates
                
                # if target_3 is starting this frame...
                if target_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_3.frameNStart = frameN  # exact frame index
                    target_3.tStart = t  # local t and not account for scr refresh
                    target_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_3.started')
                    # update status
                    target_3.status = STARTED
                    target_3.setAutoDraw(True)
                
                # if target_3 is active this frame...
                if target_3.status == STARTED:
                    # update params
                    target_3.setFillColor(target_3_color, log=False)
                    target_3.setLineColor(target_3_color, log=False)
                
                # *target_4* updates
                
                # if target_4 is starting this frame...
                if target_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_4.frameNStart = frameN  # exact frame index
                    target_4.tStart = t  # local t and not account for scr refresh
                    target_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_4.started')
                    # update status
                    target_4.status = STARTED
                    target_4.setAutoDraw(True)
                
                # if target_4 is active this frame...
                if target_4.status == STARTED:
                    # update params
                    target_4.setFillColor(target_4_color, log=False)
                    target_4.setLineColor(target_4_color, log=False)
                
                # *target_5* updates
                
                # if target_5 is starting this frame...
                if target_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    target_5.frameNStart = frameN  # exact frame index
                    target_5.tStart = t  # local t and not account for scr refresh
                    target_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(target_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_5.started')
                    # update status
                    target_5.status = STARTED
                    target_5.setAutoDraw(True)
                
                # if target_5 is active this frame...
                if target_5.status == STARTED:
                    # update params
                    target_5.setFillColor(target_5_color, log=False)
                    target_5.setLineColor(target_5_color, log=False)
                # *mouse* updates
                
                # if mouse is starting this frame...
                if mouse.status == NOT_STARTED and t >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse.frameNStart = frameN  # exact frame index
                    mouse.tStart = t  # local t and not account for scr refresh
                    mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('mouse.started', t)
                    # update status
                    mouse.status = STARTED
                    mouse.mouseClock.reset()
                    prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
                if mouse.status == STARTED:  # only update if started and not finished!
                    buttons = mouse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = environmenttools.getFromNames([start, target_1, target_2, target_3, target_4, target_5], namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(mouse):
                                    gotValidClick = True
                                    mouse.clicked_name.append(obj.name)
                            if not gotValidClick:
                                mouse.clicked_name.append(None)
                            # check whether click was in correct object
                            if gotValidClick:
                                _corr = 0
                                _corrAns = environmenttools.getFromNames([start, target_1, target_2, target_3, target_4, target_5], namespace=locals())
                                for obj in _corrAns:
                                    # is this object clicked on?
                                    if obj.contains(mouse):
                                        _corr = 1
                                mouse.corr.append(_corr)
                            x, y = mouse.getPos()
                            mouse.x.append(x)
                            mouse.y.append(y)
                            buttons = mouse.getPressed()
                            mouse.leftButton.append(buttons[0])
                            mouse.midButton.append(buttons[1])
                            mouse.rightButton.append(buttons[2])
                            mouse.time.append(mouse.mouseClock.getTime())
                
                # *cursor* updates
                
                # if cursor is starting this frame...
                if cursor.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    cursor.frameNStart = frameN  # exact frame index
                    cursor.tStart = t  # local t and not account for scr refresh
                    cursor.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cursor, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cursor.started')
                    # update status
                    cursor.status = STARTED
                    cursor.setAutoDraw(True)
                
                # if cursor is active this frame...
                if cursor.status == STARTED:
                    # update params
                    cursor.setFillColor(cursor_color, log=False)
                    cursor.setPos((cursor_x, cursor_y), log=False)
                    cursor.setLineColor(cursor_color, log=False)
                
                # *space_resp* updates
                
                # if space_resp is starting this frame...
                if space_resp.status == NOT_STARTED and t >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    space_resp.frameNStart = frameN  # exact frame index
                    space_resp.tStart = t  # local t and not account for scr refresh
                    space_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(space_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('space_resp.started', t)
                    # update status
                    space_resp.status = STARTED
                    # keyboard checking is just starting
                    space_resp.clock.reset()  # now t=0
                    space_resp.clearEvents(eventType='keyboard')
                if space_resp.status == STARTED:
                    theseKeys = space_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _space_resp_allKeys.extend(theseKeys)
                    if len(_space_resp_allKeys):
                        space_resp.keys = [key.name for key in _space_resp_allKeys]  # storing all keys
                        space_resp.rt = [key.rt for key in _space_resp_allKeys]
                        space_resp.duration = [key.duration for key in _space_resp_allKeys]
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['right'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                        key_resp.rt = _key_resp_allKeys[-1].rt
                        key_resp.duration = _key_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=trial,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # Run 'End Routine' code from trial_code
            trial_n += 1
            
            # Define the expected sequence of target centers
            target_centers = [start.pos, target_1.pos,
                              start.pos, target_2.pos,
                              start.pos, target_3.pos,
                              start.pos, target_4.pos,
                              start.pos, target_5.pos,
                              start.pos]
            
            # Error distances for each tap
            error_distances = []
            
            for idx in range(len(tap_positions)):
                tap_pos = np.array(tap_positions[idx])
                
                # Determine the expected target for this tap
                if idx < len(target_centers):
                    target_center = np.array(target_centers[idx])
                else:
                    target_center = np.array([start.pos[0], start.pos[1]])  # fallback
                
                # If the tap was on a valid target, error is 0
                if tapped_names[idx] is not None:
                    error = 0
                else:
                    # Compute Euclidean distance to expected target center
                    error = np.linalg.norm(tap_pos - target_center)
                
                error_distances.append(error)
            
            errors_np = np.array(error_distances, dtype=float)
            # number of errors
            n_errors = int(np.sum(errors_np > 0.0))
            # number of correct taps
            n_correct = key_presses - n_errors  # 11 - errors
            
            # Mean error
            if n_errors > 0:
                mean_error = round(float(np.mean(errors_np[errors_np > 0.0]))*100, 2)
            else:
                mean_error = np.nan
            
            # Save the entire list as a string in the experiment data
            thisExp.addData('error_distances', str(error_distances))
            
            # Compute times
            # 1) Convert lists to NumPy arrays
            tap_times = np.array(tap_times, dtype=float)
            space_times = np.array(space_resp.rt, dtype=float)
            
            # 2) Determine how many pairs we can compare
            n_pairs = min(len(tap_times), len(space_times))
            
            # 3) Compute pairwise differences: (tap_time - space_time)
            time_diffs = []
            for i in range(n_pairs):
                try:
                    diff = tap_times[i] - space_times[i]
                except Exception:
                    diff = np.nan
                time_diffs.append(diff)
            
            # 4) Compute average difference (mean of all pairwise diffs, ignoring NaNs)
            if len(time_diffs) > 0:
                mean_diff = float(np.nanmean(time_diffs)) * 1000
            else:
                mean_diff = np.nan
            
            # 5) Count how many taps and space presses we recorded
            n_taps = len(tap_times)
            n_spaces = len(space_times)
            
            # 6) Compute total duration: last minus first
            if len(tap_times) >= 2:
                tap_duration = float(tap_times[-1] - tap_times[0])
            else:
                tap_duration = np.nan
            
            if len(space_times) >= 2:
                # total duration
                space_duration = float(space_times[-1] - space_times[0])
                # calculate duration for every "reach" to target (not get back)
                move_durations = []
                for i in range(0, len(space_times) - 1, 2):  # jumps of 2
                    start_mov = space_times[i]
                    end_mov = space_times[i + 1]
                    duration = float(end_mov - start_mov)
                    move_durations.append(duration)
            else:
                space_duration = np.nan
                move_durations = [np.nan, np.nan, np.nan, np.nan, np.nan]
            # add them to the big lists
            movement_times.extend(move_durations)
            index_difficulties.extend(ID)
            
            # 7) round values to display nicely if needed
            mean_diff = round(mean_diff, 0)
            tap_duration = round(tap_duration, 3)
            space_duration = round(space_duration, 3)
            
            # Vectors that may be used for processing
            space_durations.append(space_duration)
            target_diameters.append(target_diameter)
            
            
            # Save to output
            thisExp.addData('tap_duration', tap_duration)
            thisExp.addData('space_duration', space_duration)
            thisExp.addData('move_durations', move_durations)
            thisExp.addData('mean_tap_space_diff', mean_diff)
            thisExp.addData('n_taps', n_taps)
            thisExp.addData('n_spaces', n_spaces)
            
            # store data for trials_loop (TrialHandler)
            trials_loop.addData('mouse.x', mouse.x)
            trials_loop.addData('mouse.y', mouse.y)
            trials_loop.addData('mouse.leftButton', mouse.leftButton)
            trials_loop.addData('mouse.midButton', mouse.midButton)
            trials_loop.addData('mouse.rightButton', mouse.rightButton)
            trials_loop.addData('mouse.time', mouse.time)
            trials_loop.addData('mouse.corr', mouse.corr)
            trials_loop.addData('mouse.clicked_name', mouse.clicked_name)
            # check responses
            if space_resp.keys in ['', [], None]:  # No response was made
                space_resp.keys = None
            trials_loop.addData('space_resp.keys',space_resp.keys)
            if space_resp.keys != None:  # we had a response
                trials_loop.addData('space_resp.rt', space_resp.rt)
                trials_loop.addData('space_resp.duration', space_resp.duration)
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
            trials_loop.addData('key_resp.keys',key_resp.keys)
            if key_resp.keys != None:  # we had a response
                trials_loop.addData('key_resp.rt', key_resp.rt)
                trials_loop.addData('key_resp.duration', key_resp.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "blank" ---
            # create an object to store info about Routine blank
            blank = data.Routine(
                name='blank',
                components=[],
            )
            blank.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for blank
            blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            blank.tStart = globalClock.getTime(format='float')
            blank.status = STARTED
            thisExp.addData('blank.started', blank.tStart)
            blank.maxDuration = 1
            # keep track of which components have finished
            blankComponents = blank.components
            for thisComponent in blank.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "blank" ---
            blank.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_loop, 'status') and thisTrials_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > blank.maxDuration-frameTolerance:
                    blank.maxDurationReached = True
                    continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=blank,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    blank.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank" ---
            for thisComponent in blank.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for blank
            blank.tStop = globalClock.getTime(format='float')
            blank.tStopRefresh = tThisFlipGlobal
            thisExp.addData('blank.stopped', blank.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if blank.maxDurationReached:
                routineTimer.addTime(-blank.maxDuration)
            elif blank.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "fb" ---
            # create an object to store info about Routine fb
            fb = data.Routine(
                name='fb',
                components=[trial_text, fb_text_time, fb_text_taps, fb_adv_text, adv_fb_resp],
            )
            fb.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from feedback_code
            # Only provide feedback if we're in the practice block
            if feedback == True and inst_file == "instructions_execution.xlsx":
                # show both feedback based on time and tap accuracy
                fb_time_x = 0
                fb_time_y = 0
                fb_taps_x = 0
                fb_taps_y = -0.2
            elif feedback == True and inst_file == "instructions_imagery.xlsx":
                # only show feedback on time
                fb_time_x = 0
                fb_time_y = 0
                fb_taps_x = -2 # set to very large value which is out of screen
                fb_taps_y = -2
            elif feedback == False:
                # not show any feedback
                fb_time_x = -2
                fb_time_y = -2
                fb_taps_x = -2
                fb_taps_y = -2
            trial_text.reset()
            trial_text.setText(trial_msg + ' ' + str(trial_n) + ' ' + out_of_msg + ' ' + str(trials_block))
            fb_text_time.reset()
            fb_text_time.setPos((fb_time_x, fb_time_y))
            fb_text_time.setText(time_msg + '\n\n' + str(round(space_duration, 2)) + ' ' + second_msg)
            fb_text_taps.reset()
            fb_text_taps.setPos((fb_taps_x, fb_taps_y))
            fb_text_taps.setText(taps_msg + '\n\n' + str(n_correct) + ' ' + out_of_msg + ' ' + str(key_presses))
            fb_adv_text.setText(adv_msg)
            # create starting attributes for adv_fb_resp
            adv_fb_resp.keys = []
            adv_fb_resp.rt = []
            _adv_fb_resp_allKeys = []
            # store start times for fb
            fb.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fb.tStart = globalClock.getTime(format='float')
            fb.status = STARTED
            thisExp.addData('fb.started', fb.tStart)
            fb.maxDuration = None
            # keep track of which components have finished
            fbComponents = fb.components
            for thisComponent in fb.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fb" ---
            fb.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_loop, 'status') and thisTrials_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *trial_text* updates
                
                # if trial_text is starting this frame...
                if trial_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_text.frameNStart = frameN  # exact frame index
                    trial_text.tStart = t  # local t and not account for scr refresh
                    trial_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_text.started')
                    # update status
                    trial_text.status = STARTED
                    trial_text.setAutoDraw(True)
                
                # if trial_text is active this frame...
                if trial_text.status == STARTED:
                    # update params
                    pass
                
                # *fb_text_time* updates
                
                # if fb_text_time is starting this frame...
                if fb_text_time.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fb_text_time.frameNStart = frameN  # exact frame index
                    fb_text_time.tStart = t  # local t and not account for scr refresh
                    fb_text_time.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fb_text_time, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fb_text_time.started')
                    # update status
                    fb_text_time.status = STARTED
                    fb_text_time.setAutoDraw(True)
                
                # if fb_text_time is active this frame...
                if fb_text_time.status == STARTED:
                    # update params
                    pass
                
                # *fb_text_taps* updates
                
                # if fb_text_taps is starting this frame...
                if fb_text_taps.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fb_text_taps.frameNStart = frameN  # exact frame index
                    fb_text_taps.tStart = t  # local t and not account for scr refresh
                    fb_text_taps.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fb_text_taps, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fb_text_taps.started')
                    # update status
                    fb_text_taps.status = STARTED
                    fb_text_taps.setAutoDraw(True)
                
                # if fb_text_taps is active this frame...
                if fb_text_taps.status == STARTED:
                    # update params
                    pass
                
                # *fb_adv_text* updates
                
                # if fb_adv_text is starting this frame...
                if fb_adv_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    fb_adv_text.frameNStart = frameN  # exact frame index
                    fb_adv_text.tStart = t  # local t and not account for scr refresh
                    fb_adv_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fb_adv_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    fb_adv_text.status = STARTED
                    fb_adv_text.setAutoDraw(True)
                
                # if fb_adv_text is active this frame...
                if fb_adv_text.status == STARTED:
                    # update params
                    pass
                
                # *adv_fb_resp* updates
                waitOnFlip = False
                
                # if adv_fb_resp is starting this frame...
                if adv_fb_resp.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    adv_fb_resp.frameNStart = frameN  # exact frame index
                    adv_fb_resp.tStart = t  # local t and not account for scr refresh
                    adv_fb_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(adv_fb_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'adv_fb_resp.started')
                    # update status
                    adv_fb_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(adv_fb_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(adv_fb_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if adv_fb_resp.status == STARTED and not waitOnFlip:
                    theseKeys = adv_fb_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _adv_fb_resp_allKeys.extend(theseKeys)
                    if len(_adv_fb_resp_allKeys):
                        adv_fb_resp.keys = _adv_fb_resp_allKeys[-1].name  # just the last key pressed
                        adv_fb_resp.rt = _adv_fb_resp_allKeys[-1].rt
                        adv_fb_resp.duration = _adv_fb_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=fb,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    fb.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fb.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fb" ---
            for thisComponent in fb.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fb
            fb.tStop = globalClock.getTime(format='float')
            fb.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fb.stopped', fb.tStop)
            # check responses
            if adv_fb_resp.keys in ['', [], None]:  # No response was made
                adv_fb_resp.keys = None
            trials_loop.addData('adv_fb_resp.keys',adv_fb_resp.keys)
            if adv_fb_resp.keys != None:  # we had a response
                trials_loop.addData('adv_fb_resp.rt', adv_fb_resp.rt)
                trials_loop.addData('adv_fb_resp.duration', adv_fb_resp.duration)
            # the Routine "fb" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisTrials_loop as finished
            if hasattr(thisTrials_loop, 'status'):
                thisTrials_loop.status = FINISHED
            # if awaiting a pause, pause now
            if trials_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trials_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed reps_per_block repeats of 'trials_loop'
        trials_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        back_forth_loop = data.TrialHandler2(
            name='back_forth_loop',
            nReps=show_back_forth, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(back_forth_loop)  # add the loop to the experiment
        thisBack_forth_loop = back_forth_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisBack_forth_loop.rgb)
        if thisBack_forth_loop != None:
            for paramName in thisBack_forth_loop:
                globals()[paramName] = thisBack_forth_loop[paramName]
        
        for thisBack_forth_loop in back_forth_loop:
            back_forth_loop.status = STARTED
            if hasattr(thisBack_forth_loop, 'status'):
                thisBack_forth_loop.status = STARTED
            currentLoop = back_forth_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisBack_forth_loop.rgb)
            if thisBack_forth_loop != None:
                for paramName in thisBack_forth_loop:
                    globals()[paramName] = thisBack_forth_loop[paramName]
            
            # --- Prepare to start Routine "controller" ---
            # create an object to store info about Routine controller
            controller = data.Routine(
                name='controller',
                components=[],
            )
            controller.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from controller_code
            if button_pressed == "next":
                cur_row += 1
            elif button_pressed == "back":
                cur_row -= 1
                
            button_pressed = "none"
            
            if cur_row < 0:
                cur_row = 0
            
            if cur_row > max_slides:
                back_forth_loop.finished = True
                show_self_assess = 0
                cur_row = max_slides
            # store start times for controller
            controller.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            controller.tStart = globalClock.getTime(format='float')
            controller.status = STARTED
            thisExp.addData('controller.started', controller.tStart)
            controller.maxDuration = None
            # keep track of which components have finished
            controllerComponents = controller.components
            for thisComponent in controller.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "controller" ---
            controller.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisBack_forth_loop, 'status') and thisBack_forth_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=controller,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    controller.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in controller.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "controller" ---
            for thisComponent in controller.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for controller
            controller.tStop = globalClock.getTime(format='float')
            controller.tStopRefresh = tThisFlipGlobal
            thisExp.addData('controller.stopped', controller.tStop)
            # the Routine "controller" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            self_assess_loop = data.TrialHandler2(
                name='self_assess_loop',
                nReps=show_self_assess, 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=data.importConditions(
                'self_assess_questions.xlsx', 
                selection=str(cur_row)
            )
            , 
                seed=None, 
            )
            thisExp.addLoop(self_assess_loop)  # add the loop to the experiment
            thisSelf_assess_loop = self_assess_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisSelf_assess_loop.rgb)
            if thisSelf_assess_loop != None:
                for paramName in thisSelf_assess_loop:
                    globals()[paramName] = thisSelf_assess_loop[paramName]
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            
            for thisSelf_assess_loop in self_assess_loop:
                self_assess_loop.status = STARTED
                if hasattr(thisSelf_assess_loop, 'status'):
                    thisSelf_assess_loop.status = STARTED
                currentLoop = self_assess_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                # abbreviate parameter names if possible (e.g. rgb = thisSelf_assess_loop.rgb)
                if thisSelf_assess_loop != None:
                    for paramName in thisSelf_assess_loop:
                        globals()[paramName] = thisSelf_assess_loop[paramName]
                
                # --- Prepare to start Routine "self_assess" ---
                # create an object to store info about Routine self_assess
                self_assess = data.Routine(
                    name='self_assess',
                    components=[imagery_icon, self_assess_text, slider, label_0, label_10, next_button, back_button, double_tap_text],
                )
                self_assess.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from self_assess_code
                win.mouseVisible = True
                back_pressed = False
                # fetch language localisation from excel sheet
                try:
                    self_assess_question = eval(f"self_assess_question_{lang_code}")
                    label_min = eval(f"label_min_{lang_code}")
                    label_max = eval(f"label_max_{lang_code}")
                except NameError:
                    self_assess_question = self_assess_question_EN
                    label_min = label_min_EN
                    label_max = label_max_EN
                
                self_assess_text.reset()
                self_assess_text.setText(self_assess_question)
                slider.reset()
                label_0.reset()
                label_0.setText(label_min)
                label_10.reset()
                label_10.setText(label_max)
                next_button.setText(next_msg)
                # reset next_button to account for continued clicks & clear times on/off
                next_button.reset()
                back_button.setText(back_msg)
                # reset back_button to account for continued clicks & clear times on/off
                back_button.reset()
                double_tap_text.setText(double_tap_msg)
                # store start times for self_assess
                self_assess.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                self_assess.tStart = globalClock.getTime(format='float')
                self_assess.status = STARTED
                thisExp.addData('self_assess.started', self_assess.tStart)
                self_assess.maxDuration = None
                # keep track of which components have finished
                self_assessComponents = self_assess.components
                for thisComponent in self_assess.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "self_assess" ---
                self_assess.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # if trial has changed, end Routine now
                    if hasattr(thisSelf_assess_loop, 'status') and thisSelf_assess_loop.status == STOPPING:
                        continueRoutine = False
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *imagery_icon* updates
                    
                    # if imagery_icon is starting this frame...
                    if imagery_icon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        imagery_icon.frameNStart = frameN  # exact frame index
                        imagery_icon.tStart = t  # local t and not account for scr refresh
                        imagery_icon.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(imagery_icon, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'imagery_icon.started')
                        # update status
                        imagery_icon.status = STARTED
                        imagery_icon.setAutoDraw(True)
                    
                    # if imagery_icon is active this frame...
                    if imagery_icon.status == STARTED:
                        # update params
                        pass
                    
                    # *self_assess_text* updates
                    
                    # if self_assess_text is starting this frame...
                    if self_assess_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        self_assess_text.frameNStart = frameN  # exact frame index
                        self_assess_text.tStart = t  # local t and not account for scr refresh
                        self_assess_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(self_assess_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'self_assess_text.started')
                        # update status
                        self_assess_text.status = STARTED
                        self_assess_text.setAutoDraw(True)
                    
                    # if self_assess_text is active this frame...
                    if self_assess_text.status == STARTED:
                        # update params
                        pass
                    
                    # *slider* updates
                    
                    # if slider is starting this frame...
                    if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        slider.frameNStart = frameN  # exact frame index
                        slider.tStart = t  # local t and not account for scr refresh
                        slider.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'slider.started')
                        # update status
                        slider.status = STARTED
                        slider.setAutoDraw(True)
                    
                    # if slider is active this frame...
                    if slider.status == STARTED:
                        # update params
                        pass
                    
                    # *label_0* updates
                    
                    # if label_0 is starting this frame...
                    if label_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        label_0.frameNStart = frameN  # exact frame index
                        label_0.tStart = t  # local t and not account for scr refresh
                        label_0.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(label_0, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'label_0.started')
                        # update status
                        label_0.status = STARTED
                        label_0.setAutoDraw(True)
                    
                    # if label_0 is active this frame...
                    if label_0.status == STARTED:
                        # update params
                        pass
                    
                    # *label_10* updates
                    
                    # if label_10 is starting this frame...
                    if label_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        label_10.frameNStart = frameN  # exact frame index
                        label_10.tStart = t  # local t and not account for scr refresh
                        label_10.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(label_10, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'label_10.started')
                        # update status
                        label_10.status = STARTED
                        label_10.setAutoDraw(True)
                    
                    # if label_10 is active this frame...
                    if label_10.status == STARTED:
                        # update params
                        pass
                    # *next_button* updates
                    
                    # if next_button is starting this frame...
                    if next_button.status == NOT_STARTED and slider.rating:
                        # keep track of start time/frame for later
                        next_button.frameNStart = frameN  # exact frame index
                        next_button.tStart = t  # local t and not account for scr refresh
                        next_button.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(next_button, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'next_button.started')
                        # update status
                        next_button.status = STARTED
                        win.callOnFlip(next_button.buttonClock.reset)
                        next_button.setAutoDraw(True)
                    
                    # if next_button is active this frame...
                    if next_button.status == STARTED:
                        # update params
                        pass
                        # check whether next_button has been pressed
                        if next_button.isClicked:
                            if not next_button.wasClicked:
                                # if this is a new click, store time of first click and clicked until
                                next_button.timesOn.append(routineTimer.getTime())
                                next_button.timesOff.append(routineTimer.getTime())
                            elif len(next_button.timesOff):
                                # if click is continuing from last frame, update time of clicked until
                                next_button.timesOff[-1] = routineTimer.getTime()
                            if not next_button.wasClicked:
                                # end routine when next_button is clicked
                                continueRoutine = False
                            if not next_button.wasClicked:
                                # run callback code when next_button is clicked
                                button_pressed = "next"
                    # take note of whether next_button was clicked, so that next frame we know if clicks are new
                    next_button.wasClicked = next_button.isClicked and next_button.status == STARTED
                    # *back_button* updates
                    
                    # if back_button is starting this frame...
                    if back_button.status == NOT_STARTED and slider.rating:
                        # keep track of start time/frame for later
                        back_button.frameNStart = frameN  # exact frame index
                        back_button.tStart = t  # local t and not account for scr refresh
                        back_button.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(back_button, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'back_button.started')
                        # update status
                        back_button.status = STARTED
                        win.callOnFlip(back_button.buttonClock.reset)
                        back_button.setAutoDraw(True)
                    
                    # if back_button is active this frame...
                    if back_button.status == STARTED:
                        # update params
                        pass
                        # check whether back_button has been pressed
                        if back_button.isClicked:
                            if not back_button.wasClicked:
                                # if this is a new click, store time of first click and clicked until
                                back_button.timesOn.append(routineTimer.getTime())
                                back_button.timesOff.append(routineTimer.getTime())
                            elif len(back_button.timesOff):
                                # if click is continuing from last frame, update time of clicked until
                                back_button.timesOff[-1] = routineTimer.getTime()
                            if not back_button.wasClicked:
                                # end routine when back_button is clicked
                                continueRoutine = False
                            if not back_button.wasClicked:
                                # run callback code when back_button is clicked
                                button_pressed = "back"
                    # take note of whether back_button was clicked, so that next frame we know if clicks are new
                    back_button.wasClicked = back_button.isClicked and back_button.status == STARTED
                    
                    # *double_tap_text* updates
                    
                    # if double_tap_text is starting this frame...
                    if double_tap_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        double_tap_text.frameNStart = frameN  # exact frame index
                        double_tap_text.tStart = t  # local t and not account for scr refresh
                        double_tap_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(double_tap_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        double_tap_text.status = STARTED
                        double_tap_text.setAutoDraw(True)
                    
                    # if double_tap_text is active this frame...
                    if double_tap_text.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer, globalClock], 
                            currentRoutine=self_assess,
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        self_assess.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in self_assess.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "self_assess" ---
                for thisComponent in self_assess.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for self_assess
                self_assess.tStop = globalClock.getTime(format='float')
                self_assess.tStopRefresh = tThisFlipGlobal
                thisExp.addData('self_assess.stopped', self_assess.tStop)
                self_assess_loop.addData('slider.response', slider.getRating())
                self_assess_loop.addData('slider.rt', slider.getRT())
                self_assess_loop.addData('next_button.numClicks', next_button.numClicks)
                if next_button.numClicks:
                   self_assess_loop.addData('next_button.timesOn', next_button.timesOn)
                   self_assess_loop.addData('next_button.timesOff', next_button.timesOff)
                else:
                   self_assess_loop.addData('next_button.timesOn', "")
                   self_assess_loop.addData('next_button.timesOff', "")
                self_assess_loop.addData('back_button.numClicks', back_button.numClicks)
                if back_button.numClicks:
                   self_assess_loop.addData('back_button.timesOn', back_button.timesOn)
                   self_assess_loop.addData('back_button.timesOff', back_button.timesOff)
                else:
                   self_assess_loop.addData('back_button.timesOn', "")
                   self_assess_loop.addData('back_button.timesOff', "")
                # the Routine "self_assess" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                # mark thisSelf_assess_loop as finished
                if hasattr(thisSelf_assess_loop, 'status'):
                    thisSelf_assess_loop.status = FINISHED
                # if awaiting a pause, pause now
                if self_assess_loop.status == PAUSED:
                    thisExp.status = PAUSED
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[globalClock], 
                    )
                    # once done pausing, restore running status
                    self_assess_loop.status = STARTED
                thisExp.nextEntry()
                
            # completed show_self_assess repeats of 'self_assess_loop'
            self_assess_loop.status = FINISHED
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # mark thisBack_forth_loop as finished
            if hasattr(thisBack_forth_loop, 'status'):
                thisBack_forth_loop.status = FINISHED
            # if awaiting a pause, pause now
            if back_forth_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                back_forth_loop.status = STARTED
        # completed show_back_forth repeats of 'back_forth_loop'
        back_forth_loop.status = FINISHED
        
        # mark thisBlocks_loop as finished
        if hasattr(thisBlocks_loop, 'status'):
            thisBlocks_loop.status = FINISHED
        # if awaiting a pause, pause now
        if blocks_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            blocks_loop.status = STARTED
    # completed n_blocks repeats of 'blocks_loop'
    blocks_loop.status = FINISHED
    
    
    # --- Prepare to start Routine "processing" ---
    # create an object to store info about Routine processing
    processing = data.Routine(
        name='processing',
        components=[],
    )
    processing.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from processing_code
    import matplotlib.pyplot as plt
    import numpy as np
    # Convert lists to arrays
    x = np.array(index_difficulties)
    y = np.array(movement_times)  # we can do y = log(y) if we want log-transformed
    # How many practice trials and main block trials do we have?
    practice_block_trials = practice_reps * n_IDs * n_targets
    main_block_trials = main_blocks_reps * n_IDs * n_targets
    # Remove practice trials based on order
    x_exe = x_ima = y_exe = y_ima = None
    if order == "0":  # Execution first
        x_exe = x[practice_block_trials : practice_block_trials + main_block_trials]
        y_exe = y[practice_block_trials : practice_block_trials + main_block_trials]
        x_ima = x[2 * practice_block_trials + main_block_trials :]
        y_ima = y[2 * practice_block_trials + main_block_trials :]
    elif order == "1":  # Imagery first
        x_ima = x[practice_block_trials : practice_block_trials + main_block_trials]
        y_ima = y[practice_block_trials : practice_block_trials + main_block_trials]
        x_exe = x[2 * practice_block_trials + main_block_trials :]
        y_exe = y[2 * practice_block_trials + main_block_trials :]
    # ---------- EXECUTION ----------
    # Lineal fit
    slope_exe, intercept_exe = np.polyfit(x_exe, y_exe, 1)
    pred_exe = slope_exe * x_exe + intercept_exe
    # Real (raw) data
    plt.scatter(x_exe, y_exe, color='blue', label='Execution block')
    # Fit line
    plt.plot(x_exe, pred_exe, color='blue', linestyle='-', 
             label=f'Fit: y={intercept_exe:.2f}+{slope_exe:.2f}x')
    # Averages by Index of Difficulty
    unique_ids_exe = np.unique(x_exe)
    means_exe = [np.mean(y_exe[x_exe == i]) for i in unique_ids_exe]
    # Line for raw data
    plt.plot(unique_ids_exe, means_exe, color='blue', linestyle='--', marker='s', label='Raw data')
    # ---------- IMAGERY ----------
    # Lineal fit
    slope_ima, intercept_ima = np.polyfit(x_ima, y_ima, 1)
    pred_ima = slope_ima * x_ima + intercept_ima
    # Real (raw) data
    plt.scatter(x_ima, y_ima, color='orange', label='Imagery block')
    # Fit line
    plt.plot(x_ima, pred_ima, color='orange', linestyle='-', 
             label=f'Fit: y={intercept_ima:.2f}+{slope_ima:.2f}x')
    # Averages by ID
    unique_ids_ima = np.unique(x_ima)
    means_ima = [np.mean(y_ima[x_ima == i]) for i in unique_ids_ima]
    # Line for raw data
    plt.plot(unique_ids_ima, means_ima, color='orange', linestyle='--', marker='s', label='Raw data')
    # ---------- Plot style ----------
    plt.xlabel("Index of Difficulty (ID)")
    plt.ylabel("Forward reaching time (s)")
    plt.title(f"Participant {expInfo['participant']} - Execution and Imagery results")
    plt.legend()
    plt.tight_layout()
    # Save plot
    plt.savefig(f"plots/regression_plot_{expInfo['participant']}.png")
    # ---------- Save to output ------------
    # Save coefficients to output
    thisExp.addData("slope_exe", slope_exe)
    thisExp.addData("intercept_exe", intercept_exe)
    thisExp.addData("slope_ima", slope_ima)
    thisExp.addData("intercept_ima", intercept_ima)
    
    # store start times for processing
    processing.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    processing.tStart = globalClock.getTime(format='float')
    processing.status = STARTED
    processing.maxDuration = None
    # keep track of which components have finished
    processingComponents = processing.components
    for thisComponent in processing.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "processing" ---
    processing.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=processing,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            processing.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in processing.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "processing" ---
    for thisComponent in processing.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for processing
    processing.tStop = globalClock.getTime(format='float')
    processing.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "processing" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "bye" ---
    # create an object to store info about Routine bye
    bye = data.Routine(
        name='bye',
        components=[bye_text, regression_image, by_resp],
    )
    bye.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from bye_code
    regression_plot = f"plots/regression_plot_{expInfo['participant']}.png"
    bye_text.setText(bye_msg)
    regression_image.setImage(regression_plot)
    # create starting attributes for by_resp
    by_resp.keys = []
    by_resp.rt = []
    _by_resp_allKeys = []
    # store start times for bye
    bye.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    bye.tStart = globalClock.getTime(format='float')
    bye.status = STARTED
    thisExp.addData('bye.started', bye.tStart)
    bye.maxDuration = None
    # keep track of which components have finished
    byeComponents = bye.components
    for thisComponent in bye.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "bye" ---
    bye.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *bye_text* updates
        
        # if bye_text is starting this frame...
        if bye_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            bye_text.frameNStart = frameN  # exact frame index
            bye_text.tStart = t  # local t and not account for scr refresh
            bye_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(bye_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'bye_text.started')
            # update status
            bye_text.status = STARTED
            bye_text.setAutoDraw(True)
        
        # if bye_text is active this frame...
        if bye_text.status == STARTED:
            # update params
            pass
        
        # *regression_image* updates
        
        # if regression_image is starting this frame...
        if regression_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            regression_image.frameNStart = frameN  # exact frame index
            regression_image.tStart = t  # local t and not account for scr refresh
            regression_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(regression_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'regression_image.started')
            # update status
            regression_image.status = STARTED
            regression_image.setAutoDraw(True)
        
        # if regression_image is active this frame...
        if regression_image.status == STARTED:
            # update params
            pass
        
        # *by_resp* updates
        waitOnFlip = False
        
        # if by_resp is starting this frame...
        if by_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            by_resp.frameNStart = frameN  # exact frame index
            by_resp.tStart = t  # local t and not account for scr refresh
            by_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(by_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'by_resp.started')
            # update status
            by_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(by_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(by_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if by_resp.status == STARTED and not waitOnFlip:
            theseKeys = by_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _by_resp_allKeys.extend(theseKeys)
            if len(_by_resp_allKeys):
                by_resp.keys = _by_resp_allKeys[-1].name  # just the last key pressed
                by_resp.rt = _by_resp_allKeys[-1].rt
                by_resp.duration = _by_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=bye,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            bye.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in bye.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "bye" ---
    for thisComponent in bye.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for bye
    bye.tStop = globalClock.getTime(format='float')
    bye.tStopRefresh = tThisFlipGlobal
    thisExp.addData('bye.stopped', bye.tStop)
    # check responses
    if by_resp.keys in ['', [], None]:  # No response was made
        by_resp.keys = None
    thisExp.addData('by_resp.keys',by_resp.keys)
    if by_resp.keys != None:  # we had a response
        thisExp.addData('by_resp.rt', by_resp.rt)
        thisExp.addData('by_resp.duration', by_resp.duration)
    thisExp.nextEntry()
    # the Routine "bye" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
