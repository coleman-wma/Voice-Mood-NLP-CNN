This is the first iteration of the web app, getting functionality running with bare minimum models.

From the app_dev.py file a website is rendered locally in html where the user can record some audio of their voice. This is then sent from javascript to flask so it can be hauled into a python function.

The python function sends it to the Google speech-to-text API which returns the text of what was said. Both audio and text can then be used as inputs to models.
